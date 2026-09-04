"""FastAPI main application entrypoint."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import logging
import secrets

from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.repositories.user_repo import UserRepository
from services.api.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB tables synchronously
    init_db_sync()
    # Real first-run bootstrap: if this is a brand-new database with no
    # accounts at all, create exactly one real admin account with a securely
    # random password (never a guessable default), and print it to the log
    # exactly once. No demo/company/session data is ever seeded here -- a
    # fresh deployment starts genuinely empty and is populated for real
    # through the Factory Setup flow once that first admin logs in.
    with get_sync_session() as db:
        user_repo = UserRepository(db)
        user_repo.seed_default_users()
        logger.info("Verified default user accounts: admin, engineer, operator.")
    yield


app = FastAPI(
    title="Fabric Bag Counter API",
    description="Conveyor Bag Counting and Dispatch Reconciliation System REST & Real-time API (v2)",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware: explicit origin allowlist (env-var configurable), since
# allow_origins=["*"] combined with allow_credentials=True is rejected by
# browsers anyway and is an insecure combination to request. Default covers
# where the bundled web UI (served by this same app, see web_dist_path below)
# and the Vite dev server run.
_default_cors_origins = "http://localhost:8080,http://127.0.0.1:8080,http://localhost:5173,http://127.0.0.1:5173"
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ALLOWED_ORIGINS", _default_cors_origins).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router, prefix="/api")

# Serve Web UI if static folder exists
web_dist_path = os.path.join(os.path.dirname(__file__), "..", "..", "web", "dist")
if os.path.exists(web_dist_path):
    app.mount("/", StaticFiles(directory=web_dist_path, html=True), name="static_spa")


def main() -> None:
    import ssl
    import uvicorn
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8080"))

    ssl_keyfile = os.environ.get("SSL_KEYFILE")
    ssl_certfile = os.environ.get("SSL_CERTFILE")
    ssl_ca_certs = os.environ.get("SSL_CA_CERTS")

    kwargs: dict[str, Any] = {
        "host": host,
        "port": port,
        "reload": False,
    }

    if ssl_keyfile and ssl_certfile:
        kwargs["ssl_keyfile"] = ssl_keyfile
        kwargs["ssl_certfile"] = ssl_certfile
        if ssl_ca_certs:
            kwargs["ssl_ca_certs"] = ssl_ca_certs
            cert_reqs = int(os.environ.get("SSL_CERT_REQS", str(ssl.CERT_REQUIRED)))
            kwargs["ssl_cert_reqs"] = cert_reqs
            logger.info(f"[mTLS] Mutual TLS enabled: cert={ssl_certfile}, ca={ssl_ca_certs}")
        else:
            logger.info(f"[TLS] Server TLS enabled: cert={ssl_certfile}")

    uvicorn.run("services.api.main:app", **kwargs)


if __name__ == "__main__":
    main()
