"""
Small-cap runtime settings API (data/smallcap_settings.json).

GET/PUT /api/settings and /api/settings/ — same handler (no 307 for slash variants).
POST /api/settings/reset and /api/settings/reset/ — reset to defaults.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from api.deps import invalidate_smallcap_engine_cache
from api.utils import sanitize_for_json

logger = logging.getLogger(__name__)

# Prefix on router; trailing-slash aliases avoid 307 redirects for strict clients.
router = APIRouter(prefix="/api/settings", tags=["settings"])


def _get_smallcap_settings():
    from swing_trader.small_cap.settings_config import load_settings

    s = load_settings()
    return sanitize_for_json(s.model_dump(mode="json"))


router.add_api_route("", _get_smallcap_settings, methods=["GET"], summary="Get small-cap settings")
router.add_api_route("/", _get_smallcap_settings, methods=["GET"], include_in_schema=False)


def _put_smallcap_settings(body: Dict[str, Any]):
    """
    Deep-merge JSON body into current settings, validate, save to disk.

    Unknown top-level keys cause 422 (model extra=forbid after merge).
    """
    from swing_trader.small_cap.settings_config import apply_settings_patch

    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="Body must be a JSON object")

    try:
        updated = apply_settings_patch(body)
    except ValidationError as e:
        logger.warning("Small-cap settings validation failed: %s", e)
        raise HTTPException(status_code=422, detail=e.errors()) from e
    except TypeError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    invalidate_smallcap_engine_cache()
    logger.info("Small-cap settings updated; engine cache cleared")

    return sanitize_for_json(
        {
            "ok": True,
            "settings": updated.model_dump(mode="json"),
        }
    )


router.add_api_route("", _put_smallcap_settings, methods=["PUT"], summary="Update small-cap settings (merge)")
router.add_api_route("/", _put_smallcap_settings, methods=["PUT"], include_in_schema=False)


def _reset_smallcap_settings():
    from swing_trader.small_cap.settings_config import default_settings, save_settings

    s = default_settings()
    save_settings(s)
    invalidate_smallcap_engine_cache()
    logger.info("Small-cap settings reset to defaults; engine cache cleared")
    return sanitize_for_json({"ok": True, "settings": s.model_dump(mode="json")})


router.add_api_route("/reset", _reset_smallcap_settings, methods=["POST"], summary="Reset small-cap settings to defaults")
router.add_api_route("/reset/", _reset_smallcap_settings, methods=["POST"], include_in_schema=False)
