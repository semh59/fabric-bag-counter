"""Enterprise Mutual TLS (mTLS) Certificate Authority and Key Generator (§4.4, §8.1).

Generates real, cryptographically valid X.509 Root CA, Server, and Edge Client
certificates with Subject Alternative Names (SANs) for securing Edge-to-Server
connections (PostgreSQL & FastAPI).
"""

from __future__ import annotations

import argparse
import datetime
import ipaddress
from pathlib import Path
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


def generate_private_key(key_size: int = 2048) -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
    )


def save_key(key: rsa.RSAPrivateKey, path: Path) -> None:
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path.write_bytes(pem)


def save_cert(cert: x509.Certificate, path: Path) -> None:
    pem = cert.public_bytes(serialization.Encoding.PEM)
    path.write_bytes(pem)


def generate_mtls_certificates(
    output_dir: Path,
    organization: str = "Fabric Bag Counter Industrial",
    validity_days: int = 365,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc)
    expiry = now + datetime.timedelta(days=validity_days)

    # 1. Root Certificate Authority (CA)
    ca_key = generate_private_key()
    ca_subject = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization),
        x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Security Authority"),
        x509.NameAttribute(NameOID.COMMON_NAME, "Fabric Root CA"),
    ])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_subject)
        .issuer_name(ca_subject)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=validity_days * 3))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )

    ca_key_path = output_dir / "ca.key"
    ca_cert_path = output_dir / "ca.crt"
    save_key(ca_key, ca_key_path)
    save_cert(ca_cert, ca_cert_path)

    # 2. Server Certificate (for PostgreSQL & API)
    server_key = generate_private_key()
    server_subject = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization),
        x509.NameAttribute(NameOID.COMMON_NAME, "cs-server"),
    ])
    san_names: list[x509.GeneralName] = [
        x509.DNSName("localhost"),
        x509.DNSName("host.docker.internal"),
        x509.DNSName("postgres"),
        x509.DNSName("cs-postgres"),
        x509.DNSName("cs-api"),
        x509.DNSName("api"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
    ]
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(server_subject)
        .issuer_name(ca_subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(expiry)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectAlternativeName(san_names), critical=False)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    server_key_path = output_dir / "server.key"
    server_cert_path = output_dir / "server.crt"
    save_key(server_key, server_key_path)
    save_cert(server_cert, server_cert_path)

    # 3. Edge Client Certificate (for Edge Ingest & Inference Workers)
    client_key = generate_private_key()
    client_subject = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization),
        x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Edge Fleet"),
        x509.NameAttribute(NameOID.COMMON_NAME, "cs-edge-worker-node1"),
    ])
    client_cert = (
        x509.CertificateBuilder()
        .subject_name(client_subject)
        .issuer_name(ca_subject)
        .public_key(client_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(expiry)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    client_key_path = output_dir / "edge_client.key"
    client_cert_path = output_dir / "edge_client.crt"
    save_key(client_key, client_key_path)
    save_cert(client_cert, client_cert_path)

    return {
        "ca_cert": ca_cert_path,
        "ca_key": ca_key_path,
        "server_cert": server_cert_path,
        "server_key": server_key_path,
        "client_cert": client_cert_path,
        "client_key": client_key_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mutual TLS (mTLS) certificate bundle")
    parser.add_argument("--out-dir", default="certs", help="Output directory for certificates")
    parser.add_argument("--days", type=int, default=365, help="Validity period in days")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    paths = generate_mtls_certificates(out_path, validity_days=args.days)
    print("=" * 60)
    print("  Mutual TLS (mTLS) Industrial Certificates Generated")
    print("=" * 60)
    for name, p in paths.items():
        print(f"  {name:15s} -> {p}")
    print("\nConfiguration for Edge Machine (.env):")
    print(f"  DB_SSL_MODE=verify-ca")
    print(f"  DB_SSL_CA={paths['ca_cert']}")
    print(f"  DB_SSL_CERT={paths['client_cert']}")
    print(f"  DB_SSL_KEY={paths['client_key']}")
    print("\nConfiguration for Server Stack (.env):")
    print(f"  SSL_CERTFILE={paths['server_cert']}")
    print(f"  SSL_KEYFILE={paths['server_key']}")
    print(f"  SSL_CA_CERTS={paths['ca_cert']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
