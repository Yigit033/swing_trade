"""
FastAPI Backend — Swing Trade AI Dashboard
Serves as the API layer between Next.js frontend and Python trading engine.
"""

import sys
import logging
from pathlib import Path

# Add project root to path so swing_trader package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import trades, pending, performance, lookup, scanner, genai, backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(
    title="Swing Trade AI API",
    description="AI-powered swing trading dashboard backend",
    version="2.1.0",
)

# CORS — allow Next.js dev server and production Vercel domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to specific domains in production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(trades.router,      prefix="/api/trades",      tags=["trades"])
app.include_router(pending.router,     prefix="/api/pending",     tags=["pending"])
app.include_router(performance.router, prefix="/api/performance", tags=["performance"])
app.include_router(lookup.router,      prefix="/api/lookup",      tags=["lookup"])
app.include_router(scanner.router,     prefix="/api/scanner",     tags=["scanner"])
app.include_router(genai.router,       prefix="/api/genai",       tags=["genai"])
app.include_router(backtest.router,    prefix="/api/backtest",    tags=["backtest"])


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.1.0"}
