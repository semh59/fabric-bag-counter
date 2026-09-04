"""Tests for Edge-to-Server Mutual TLS (mTLS) Security & Certificate Verification."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from packages.cs_storage.db import (
    get_async_connect_args,
    get_db_ssl_config,
    get_sync_connect_args,
)
from services.api.main import app
from services.api.mtls import get_client_certificate_info, get_tls_status, require_edge_client_cert
from tools.generate_mtls_certs import generate_mtls_certificates


def test_generate_mtls_certificates(tmp_path: Path):
    certs = generate_mtls_certificates(tmp_path, validity_days=30)
    for key in ("ca_cert", "ca_key", "server_cert", "server_key", "client_cert", "client_key"):
        assert certs[key].exists(), f"Certificate file {key} was not generated"
        assert certs[key].stat().st_size > 0

    # Verify certificate content using cryptography
    from cryptography import x509
    ca = x509.load_pem_x509_certificate(certs["ca_cert"].read_bytes())
    assert "Fabric Root CA" in ca.subject.rfc4514_string()

    server = x509.load_pem_x509_certificate(certs["server_cert"].read_bytes())
    assert "cs-server" in server.subject.rfc4514_string()
    assert server.issuer == ca.subject

    client = x509.load_pem_x509_certificate(certs["client_cert"].read_bytes())
    assert "cs-edge-worker-node1" in client.subject.rfc4514_string()
    assert client.issuer == ca.subject


def test_db_ssl_config_defaults(monkeypatch):
    monkeypatch.delenv("DB_SSL_MODE", raising=False)
    monkeypatch.delenv("DB_SSL_CA", raising=False)
    monkeypatch.delenv("DB_SSL_ROOTCERT", raising=False)
    monkeypatch.delenv("DB_SSL_CERT", raising=False)
    monkeypatch.delenv("DB_SSL_KEY", raising=False)

    cfg = get_db_ssl_config()
    assert not cfg["is_mtls_configured"]
    assert not cfg["is_ssl_enabled"]

    # SQLite returns default options
    sqlite_args = get_sync_connect_args("sqlite:///./data/test.db")
    assert sqlite_args.get("check_same_thread") is False

    async_sqlite = get_async_connect_args("sqlite+aiosqlite:///./data/test.db")
    assert async_sqlite == {}


def test_db_ssl_config_with_mtls(tmp_path: Path, monkeypatch):
    certs = generate_mtls_certificates(tmp_path, validity_days=10)

    monkeypatch.setenv("DB_SSL_MODE", "verify-full")
    monkeypatch.setenv("DB_SSL_CA", str(certs["ca_cert"]))
    monkeypatch.setenv("DB_SSL_CERT", str(certs["client_cert"]))
    monkeypatch.setenv("DB_SSL_KEY", str(certs["client_key"]))

    cfg = get_db_ssl_config()
    assert cfg["is_mtls_configured"]
    assert cfg["is_ssl_enabled"]

    # Sync connect args
    pg_url = "postgresql://user:pass@localhost:5432/db"
    sync_args = get_sync_connect_args(pg_url)
    assert sync_args["sslmode"] == "verify-full"
    assert sync_args["sslrootcert"] == str(certs["ca_cert"])
    assert sync_args["sslcert"] == str(certs["client_cert"])
    assert sync_args["sslkey"] == str(certs["client_key"])

    # Async connect args
    async_args = get_async_connect_args("postgresql+asyncpg://user:pass@localhost:5432/db")
    assert "ssl" in async_args
    ssl_ctx = async_args["ssl"]
    assert ssl_ctx is not False
    assert ssl_ctx.check_hostname is True


def test_api_mtls_proxy_headers():
    req = MagicMock()
    req.headers = {
        "X-SSL-Client-Verify": "SUCCESS",
        "X-SSL-Client-S-DN": "CN=cs-edge-worker-01,O=Fabric",
        "X-SSL-Client-SHA256": "abc123456789",
    }
    req.scope = {}
    info = get_client_certificate_info(req)
    assert info["is_verified"] is True
    assert info["source"] == "proxy_header"
    assert "cs-edge-worker-01" in info["subject_dn"]

    verified = require_edge_client_cert(req)
    assert verified["is_verified"] is True


def test_api_system_tls_status_endpoint():
    client = TestClient(app)
    res = client.get("/api/system/tls-status")
    assert res.status_code == 200
    data = res.json()
    assert "ssl_enabled" in data
    assert "mtls_enabled" in data
    assert "client_connection" in data
