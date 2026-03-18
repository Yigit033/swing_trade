"""
FastAPI Backend — Swing Trade AI Dashboard
Serves as the API layer between Next.js frontend and Python trading engine.
"""

import sys
import logging
from pathlib import Path

# Add project root to path so swing_trader package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Load .env BEFORE any other imports so os.getenv() works everywhere ──
import os as _os
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; env vars must be set externally

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import trades, pending, performance, lookup, scanner, genai, backtest, regime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Swing Trade AI API",
    description="AI-powered swing trading dashboard backend",
    version="2.1.0",
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """500 hatalarını logla; CORS header'ları ekle (500'de CORS hatası önlemi)."""
    logger.exception("Unhandled exception: %s", exc)
    origin = request.headers.get("origin")
    resp = JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )
    if origin and origin in _origins:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Credentials"] = "true"
    return resp


# CORS — credentials=true için "*" kullanılamaz, localhost açıkça eklenmeli
_cors_origins = _os.environ.get("CORS_ORIGINS", "*")
if _cors_origins == "*":
    _origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
else:
    _origins = [o.strip() for o in _cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
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
app.include_router(regime.router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.1.0"}


@app.get("/api/auth/status")
async def auth_status():
    """Auth config check — 401 debug için. CORS + secrets doğrulama."""
    import api.auth as auth_mod
    return {
        "auth_configured": bool(
            (auth_mod.SUPABASE_URL and auth_mod.SUPABASE_ANON_KEY) or auth_mod.SUPABASE_JWT_SECRET
        ),
        "has_supabase_url": bool(auth_mod.SUPABASE_URL),
        "has_jwt_secret": bool(auth_mod.SUPABASE_JWT_SECRET),
        "cors_origins_count": len(_origins),
    }
