"""
FastAPI Backend — Swing Trade AI Dashboard
Serves as the API layer between Next.js frontend and Python trading engine.
"""

import sys
import asyncio
import logging
import threading
from contextlib import asynccontextmanager, suppress
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PENDING_SCHEDULER_ENABLED = _os.environ.get("ENABLE_PENDING_SCHEDULER", "1").strip().lower() not in (
    "0", "false", "no",
)
_PENDING_INTERVAL_SEC = int(_os.environ.get("PENDING_CONFIRM_INTERVAL_SEC", "300"))

# CORS — credentials=true için "*" kullanılamaz. Varsayılanlara production
# frontend'i de dahil: secret set edilmemiş olsa bile Vercel preflight geçsin.
_DEFAULT_CORS_ORIGINS = (
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://localhost:3000",
    "https://swingtrade.vercel.app",
)

# Routers are imported after TCP bind (see lifespan). Fly's proxy waits ~8s for
# 0.0.0.0:8000; importing scanners/pandas at module load loses that race.
# threading.Event: asyncio.Event created at import binds to the wrong loop
# under TestClient / reload.
_routers_ready = threading.Event()
_routers_mounted = False
_mount_lock = threading.Lock()
_HEALTH_PATHS = frozenset({"/api/health", "/api/health/"})


def cors_allow_origins(raw: str | None = None) -> list[str]:
    """Merge explicit CORS_ORIGINS with localhost + the production Vercel origin."""
    origins = list(_DEFAULT_CORS_ORIGINS)
    text = (raw if raw is not None else _os.environ.get("CORS_ORIGINS", "")).strip()
    if not text or text == "*":
        return origins
    for part in text.split(","):
        origin = part.strip().rstrip("/")
        if origin and origin not in origins:
            origins.append(origin)
    return origins


_origins = cors_allow_origins()


def _import_router_modules():
    """Heavy import — run in a worker thread so the event loop can bind :8000."""
    from api.routers import (
        backtest,
        genai,
        lookup,
        pending,
        performance,
        regime,
        scanner,
        trades,
        settings as settings_router,
    )

    return (
        trades,
        pending,
        performance,
        lookup,
        scanner,
        genai,
        backtest,
        settings_router,
        regime,
    )


def _include_routers(app: FastAPI, modules: tuple) -> None:
    global _routers_mounted
    if _routers_mounted:
        return
    (
        trades,
        pending,
        performance,
        lookup,
        scanner,
        genai,
        backtest,
        settings_router,
        regime,
    ) = modules
    app.include_router(trades.router, prefix="/api/trades", tags=["trades"])
    app.include_router(pending.router, prefix="/api/pending", tags=["pending"])
    app.include_router(performance.router, prefix="/api/performance", tags=["performance"])
    app.include_router(lookup.router, prefix="/api/lookup", tags=["lookup"])
    app.include_router(scanner.router, prefix="/api/scanner", tags=["scanner"])
    app.include_router(genai.router, prefix="/api/genai", tags=["genai"])
    app.include_router(backtest.router, prefix="/api/backtest", tags=["backtest"])
    app.include_router(settings_router.router)
    app.include_router(regime.router)
    _routers_mounted = True


def _run_scheduled_pending_and_exits() -> tuple[int, list]:
    """Sync yfinance/Postgres work — must not run on the asyncio event loop."""
    from api.deps import get_paper_tracker

    tracker = get_paper_tracker()
    processed = tracker.confirm_pending_trades(None)
    updated = tracker.update_all_open_trades(None)
    closed = [
        t for t in (updated or [])
        if t.get("status") not in ("OPEN", "PENDING")
    ]
    return len(processed or []), closed


async def _scheduled_pending_confirm_loop() -> None:
    """Periodically run pending confirmation for all users (no UI required)."""
    await asyncio.sleep(15)
    while True:
        try:
            n, closed = await asyncio.to_thread(_run_scheduled_pending_and_exits)
            if n:
                logger.info("Scheduled pending confirm: processed %d trade(s)", n)
            if closed:
                logger.info(
                    "Scheduled exit check: %d trade(s) closed (%s)",
                    len(closed),
                    ", ".join(t.get("ticker", "?") for t in closed),
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Scheduled pending confirm failed")
        await asyncio.sleep(max(60, _PENDING_INTERVAL_SEC))


async def ensure_routers_mounted(app: FastAPI) -> None:
    """Load routers if needed. Safe from TestClient (no lifespan) and from _boot."""
    if _routers_ready.is_set():
        return
    modules = await asyncio.to_thread(_import_router_modules)
    with _mount_lock:
        if not _routers_mounted:
            _include_routers(app, modules)
        _routers_ready.set()


async def _boot(app: FastAPI) -> None:
    """Import routers off the event loop, then start background schedulers."""
    try:
        await ensure_routers_mounted(app)
        logger.info("API routers mounted (port already listening)")
    except Exception:
        logger.exception("Failed to mount API routers")
        return

    if _PENDING_SCHEDULER_ENABLED:
        t = asyncio.create_task(_scheduled_pending_confirm_loop())
        app.state.bg_tasks.append(t)
        logger.info(
            "Pending scheduler enabled (interval=%ss, disable with ENABLE_PENDING_SCHEDULER=0)",
            _PENDING_INTERVAL_SEC,
        )

    from api.auto_scan import auto_scan_loop

    t = asyncio.create_task(auto_scan_loop())
    app.state.bg_tasks.append(t)
    logger.info("Auto-scan loop started (enable/configure via settings: auto_scan.enabled)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Yield immediately so uvicorn can bind :8000 before router imports.
    # Heavy work runs in a background task (thread) after this yield.
    app.state.bg_tasks = []
    boot = asyncio.create_task(_boot(app))
    app.state.bg_tasks.append(boot)
    yield
    for task in list(app.state.bg_tasks):
        task.cancel()
    for task in list(app.state.bg_tasks):
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(
    title="Swing Trade AI API",
    description="AI-powered swing trading dashboard backend",
    version="2.1.0",
    lifespan=lifespan,
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


@app.middleware("http")
async def wait_until_routers_ready(request: Request, call_next):
    """Health answers immediately; other routes wait for deferred router import."""
    if request.url.path in _HEALTH_PATHS:
        return await call_next(request)
    await ensure_routers_mounted(request.app)
    return await call_next(request)


# Added last so it wraps the wait middleware — 503 during boot still gets CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "2.1.0",
        "routers_ready": _routers_ready.is_set(),
    }


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
        "routers_ready": _routers_ready.is_set(),
    }
