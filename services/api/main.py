"""FastAPI main application entrypoint."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.demo_seeder import seed_demo_data
from packages.cs_storage.repositories.user_repo import UserRepository
from services.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB tables synchronously
    init_db_sync()
    # Seed default roles and demo dataset
    with get_sync_session() as db:
        user_repo = UserRepository(db)
        user_repo.seed_default_users()
        seed_demo_data(db, force_reset=False)
    yield


app = FastAPI(
    title="Çuval Sayım Sistemi API",
    description="Konveyör Çuval Sayım ve Sevkiyat Mutabakat Sistemi REST & Real-time API (v2)",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    import uvicorn
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8080"))
    uvicorn.run("services.api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
