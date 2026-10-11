"""Run with python -m swing_trader.discovery; every action is explicit/opt-in."""
import argparse
import json
import logging
import pickle
from pathlib import Path

from .master import fetch_snapshot, load_snapshot, save_snapshot, write_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("snapshot")
    fetch.add_argument("--root", type=Path, default=Path("output/discovery_v2/snapshots"))
    legacy = sub.add_parser("legacy")
    legacy.add_argument("--settings", type=Path, help="Explicit sanitized SmallCapSettings patch; no DB access")
    legacy.add_argument("--output", required=True, type=Path)
    recorded = sub.add_parser("recorded-legacy", help="Adapt a sanitized scan-history JSON export, offline")
    recorded.add_argument("--input", required=True, type=Path)
    recorded.add_argument("--run-id", required=True, type=int)
    recorded.add_argument("--output", required=True, type=Path)
    comp = sub.add_parser("compare")
    comp.add_argument("--snapshot", required=True, type=Path)
    comp.add_argument("--legacy", required=True, type=Path)
    comp.add_argument("--output", required=True, type=Path)
    comp.add_argument("--trusted-cache", type=Path, help="Optional LOCAL TRUSTED pickle; pickle executes code. Never use untrusted files.")
    comp.add_argument("--as-of", help="Explicit completed historical session YYYY-MM-DD for cache inspection")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    if args.command == "snapshot":
        snapshot, raw = fetch_snapshot()
        path = save_snapshot(args.root, snapshot, raw)
        print(json.dumps({"snapshot": str(path), "summary": snapshot["summary"]}, indent=2))
    elif args.command == "legacy":
        from .legacy import collect_legacy
        patch = json.loads(args.settings.read_text(encoding="utf-8")) if args.settings else {}
        data = collect_legacy(patch)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, data)
        print(json.dumps({"complete": data["complete"], "funnel": data["funnel"], "queries": [{"label": q["label"], "status": q["status"], "count": len(q["rows"])} for q in data["queries"]]}, indent=2))
    elif args.command == "recorded-legacy":
        from .legacy import recorded_run_legacy
        runs = json.loads(args.input.read_text(encoding="utf-8"))
        matches = [r for r in runs if r["id"] == args.run_id]
        if len(matches) != 1:
            parser.error("Expected exactly one matching run")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, recorded_run_legacy(matches[0]))
        print(str(args.output))
    else:
        from .comparison import compare, inspect_cache
        snapshot = load_snapshot(args.snapshot)
        champion = json.loads(args.legacy.read_text(encoding="utf-8"))
        result = compare(snapshot, champion)
        if args.trusted_cache:
            if not args.as_of:
                parser.error("--trusted-cache requires explicit --as-of")
            from swing_trader.small_cap.settings_config import SmallCapSettings
            raw = args.trusted_cache.read_bytes()
            bars = pickle.loads(raw)
            result["cache_inspection"] = inspect_cache(snapshot, champion, bars,
                SmallCapSettings(), args.as_of, {"path": str(args.trusted_cache), "sha256": digest(raw),
                "ticker_count": len(bars), "settings": "CODE_DEFAULTS_NOT_HISTORICAL_EFFECTIVE_SETTINGS"})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
        print(json.dumps({k: v for k, v in result.items() if k not in {"candidate_not_in_legacy", "cache_inspection"}}, indent=2))
        if "cache_inspection" in result:
            print(json.dumps(result["cache_inspection"], indent=2))


if __name__ == "__main__":
    main()
