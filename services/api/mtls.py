"""Mutual TLS (mTLS) Authentication and Client Certificate Verification (§4.4, §8.1).

Verifies client certificates presented by edge worker nodes (ingest/inference)
either directly through ASGI TLS extensions or via reverse proxy headers.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)

# Environment options for API mTLS
SSL_KEYFILE = os.getenv("SSL_KEYFILE")
SSL_CERTFILE = os.getenv("SSL_CERTFILE")
SSL_CA_CERTS = os.getenv("SSL_CA_CERTS")
# REQUIRE_EDGE_MTLS: if true, endpoints protected by require_edge_client_cert will reject non-mTLS requests
REQUIRE_EDGE_MTLS = os.getenv("REQUIRE_EDGE_MTLS", "false").lower() in ("1", "true", "yes")


def get_client_certificate_info(request: Request) -> dict[str, Any]:
    """Extract client certificate details from ASGI scope or reverse proxy headers."""
    info: dict[str, Any] = {
        "is_verified": False,
        "subject_dn": None,
        "common_name": None,
        "issuer_dn": None,
        "fingerprint_sha256": None,
        "source": "none",
    }

    # 1. Check reverse proxy headers (e.g. Nginx, Traefik, Caddy, Envoy)
    proxy_verify = request.headers.get("X-SSL-Client-Verify")
    if proxy_verify:
        is_verified = proxy_verify.upper() == "SUCCESS"
        info.update({
            "is_verified": is_verified,
            "subject_dn": request.headers.get("X-SSL-Client-S-DN"),
            "common_name": request.headers.get("X-SSL-Client-CN") or request.headers.get("X-SSL-Client-S-DN"),
            "issuer_dn": request.headers.get("X-SSL-Client-I-DN"),
            "fingerprint_sha256": request.headers.get("X-SSL-Client-SHA256"),
            "source": "proxy_header",
        })
        return info

    # 2. Check direct ASGI TLS extension (if uvicorn terminated TLS with ssl_ca_certs)
    tls_ext = request.scope.get("extensions", {}).get("tls", {})
    if tls_ext:
        client_cert = tls_ext.get("client_cert_chain")
        client_cert_bytes = client_cert[0] if client_cert else None
        if client_cert_bytes:
            try:
                from cryptography import x509
                cert = x509.load_der_x509_certificate(client_cert_bytes)
                import hashlib
                fp = hashlib.sha256(client_cert_bytes).hexdigest()
                cn = cert.subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)
                common_name = cn[0].value if cn else None

                info.update({
                    "is_verified": True,
                    "subject_dn": cert.subject.rfc4514_string(),
                    "common_name": common_name,
                    "issuer_dn": cert.issuer.rfc4514_string(),
                    "fingerprint_sha256": fp,
                    "source": "asgi_tls",
                })
                return info
            except Exception as e:
                logger.warning(f"[mTLS] Failed parsing ASGI client certificate: {e}")

    # Fallback to localhost / internal bypass if mTLS is not strictly enforced
    client_host = request.client.host if request.client else "unknown"
    if client_host in ("127.0.0.1", "::1", "localhost", "testclient"):
        info["source"] = "loopback"
        info["common_name"] = "loopback-internal"
        if not REQUIRE_EDGE_MTLS:
            info["is_verified"] = True

    return info


def require_edge_client_cert(request: Request) -> dict[str, Any]:
    """FastAPI dependency to enforce that requests from edge workers come over verified mTLS."""
    info = get_client_certificate_info(request)
    if REQUIRE_EDGE_MTLS and not info["is_verified"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Client certificate required. Mutual TLS (mTLS) authentication failed or certificate not presented.",
        )
    return info


def get_tls_status() -> dict[str, Any]:
    """Report server TLS and mutual TLS configuration status."""
    return {
        "ssl_enabled": bool(SSL_CERTFILE and SSL_KEYFILE),
        "mtls_enabled": bool(SSL_CA_CERTS and SSL_CERTFILE),
        "require_edge_mtls": REQUIRE_EDGE_MTLS,
        "ca_certs_configured": bool(SSL_CA_CERTS),
        "certfile": SSL_CERTFILE if SSL_CERTFILE else None,
        "ca_file": SSL_CA_CERTS if SSL_CA_CERTS else None,
    }
