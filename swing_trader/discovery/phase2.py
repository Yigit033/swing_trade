"""Explicit research CLI: python -m swing_trader.discovery.phase2 --help.

All artifacts are exclusive/immutable. No implicit network on scan/replay/track.
"""
from __future__ import annotations
import argparse
import json
import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path

from .master import load_snapshot, write_json, utc_now, digest
from .bars import prepare, pilot_records, object_id
from .calendar import last_completed


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--snapshot", required=True, type=Path)
    plan.add_argument("--count", type=int, default=240, help="0 selects all candidates, no truncation")
    plan.add_argument("--trusted-liquidity-cache", type=Path, help="Explicit LOCAL TRUSTED pickle; sampling only")
    plan.add_argument("--output", required=True, type=Path)
    prep = sub.add_parser("prepare")
    prep.add_argument("--plan", required=True, type=Path)
    prep.add_argument("--cache", required=True, type=Path)
    prep.add_argument("--run", required=True, type=Path)
    prep.add_argument("--session", default=None)
    prep.add_argument("--previous", type=Path)
    prep.add_argument("--workers", type=int, choices=[1, 2], default=1)
    prep.add_argument("--min-interval", type=float, default=1.0, help="Global history-call pacing; minimum 0.5s")
    scan = sub.add_parser("scan")
    scan.add_argument("--snapshot", required=True, type=Path)
    scan.add_argument("--preparation", required=True, type=Path)
    scan.add_argument("--cache", required=True, type=Path)
    scan.add_argument("--settings", type=Path)
    scan.add_argument("--legacy", type=Path)
    scan.add_argument("--facts", type=Path, help="Optional sanitized PIT metadata; no implicit fetch")
    scan.add_argument("--as-of", default=None)
    scan.add_argument("--session", default=None, help="Optional earlier retrospective cutoff; NEVER a PIT universe claim")
    scan.add_argument("--output", required=True, type=Path)
    replay = sub.add_parser("replay")
    replay.add_argument("--scan", required=True, type=Path)
    replay.add_argument("--inputs", required=True, type=Path)
    replay.add_argument("--cache", required=True, type=Path)
    track = sub.add_parser("track")
    track.add_argument("--scan", required=True, type=Path)
    track.add_argument("--output", required=True, type=Path)
    track.add_argument("--previous", type=Path)
    track.add_argument("--cohort", choices=["PROSPECTIVE_SHADOW", "HISTORICAL_RESEARCH"], required=True)
    fwd = sub.add_parser("forward")
    fwd.add_argument("--opportunities", required=True, type=Path)
    fwd.add_argument("--preparation", required=True, type=Path)
    fwd.add_argument("--cache", required=True, type=Path)
    fwd.add_argument("--output", required=True, type=Path)
    facts = sub.add_parser("facts", help="Explicit optional network enrichment for current trigger positives only")
    facts.add_argument("--scan", required=True, type=Path)
    facts.add_argument("--snapshot", required=True, type=Path)
    facts.add_argument("--cache", required=True, type=Path)
    facts.add_argument("--output", required=True, type=Path)
    facts.add_argument("--limit", type=int, default=100)
    args = p.parse_args()
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    if args.command == "plan":
        snapshot = load_snapshot(args.snapshot)
        liquidity = {}
        if args.trusted_liquidity_cache:
            import pickle
            # User opt-in only; never load remote/untrusted pickle.
            with args.trusted_liquidity_cache.open("rb") as handle:
                cached = pickle.load(handle)
            for symbol, frame in cached.items():
                if frame is not None and len(frame) >= 20:
                    liquidity[symbol] = float((frame.Close.tail(20)*frame.Volume.tail(20)).mean())
        n = args.count or sum(r["discovery_status"] == "CANDIDATE" for r in snapshot["records"])
        if n < 0:
            p.error("count must be nonnegative")
        records, sampling = pilot_records(snapshot, n, liquidity)
        benchmarks = [{"symbol": s, "exchange": "BENCHMARK", "yfinance_symbol_candidate": s} for s in ("SPY", "^VIX")]
        result = {"universe_id": object_id(snapshot), "universe_path": str(args.snapshot),
                  "created_at": utc_now(), "sampling": sampling, "records": records + benchmarks,
                  "candidate_count": len(records), "benchmark_count": 2,
                  "liquidity_cache_sha256": digest(args.trusted_liquidity_cache.read_bytes()) if args.trusted_liquidity_cache else None}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
        print(json.dumps({k:v for k,v in result.items() if k != "records"}, indent=2))
    elif args.command == "prepare":
        import yfinance as yf
        args.cache.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(args.cache / "yf_internal"))
        plan = read(args.plan)
        previous = {o["symbol"]: o for o in read(args.previous)["outcomes"]} if args.previous else None
        session = date.fromisoformat(args.session) if args.session else last_completed(datetime.now(timezone.utc))
        from .bars import YahooProvider
        result = prepare(plan["records"], session, args.cache, args.run, previous=previous,
                         provider=YahooProvider(args.min_interval), workers=args.workers,
                         metadata={"plan_id": object_id(plan), "universe_id": plan["universe_id"], "sampling": plan["sampling"],
                                   "min_interval": args.min_interval})
        print(json.dumps({k:v for k,v in result.items() if k not in {"outcomes", "symbols"}}, indent=2))
    elif args.command == "scan":
        from swing_trader.small_cap.settings_config import SmallCapSettings, _prune_removed_keys
        from .shadow import scan, save_scan
        snapshot, preparation = load_snapshot(args.snapshot), read(args.preparation)
        if preparation["metadata"]["universe_id"] != object_id(snapshot):
            p.error("Universe identity differs from preparation")
        if args.session:
            if args.session > preparation["session"]:
                p.error("Retrospective cutoff cannot be after prepared session")
            preparation = {**preparation, "source_preparation_id": object_id(preparation),
                           "session": args.session, "reconstruction": "CURRENT_CACHE_RETROSPECTIVE_NOT_POINT_IN_TIME"}
        patch = read(args.settings) if args.settings else {}
        settings = SmallCapSettings.model_validate(_prune_removed_keys(patch))
        legacy, facts = read(args.legacy) if args.legacy else {}, read(args.facts) if args.facts else {}
        started = time.monotonic()
        decision_at = args.as_of or utc_now()
        result = scan(snapshot, preparation, args.cache, settings, decision_at, legacy, facts)
        elapsed = time.monotonic() - started
        # Freeze ALL inputs for offline replay, not merely paths to mutable files.
        inputs = {"snapshot": snapshot, "preparation": preparation,
                   "settings": settings.model_dump(mode="json"), "legacy": legacy, "facts": facts, "decision_at": decision_at}
        execution = {"elapsed_seconds": elapsed, "settings_provenance":
                   str(args.settings) if args.settings else "CODE_DEFAULTS_NO_DB_OVERLAY",
                   "production_removed_key_pruning_applied": _prune_removed_keys(patch) != patch}
        save_scan(args.output, result, inputs, execution)
        print(json.dumps({"metrics": result["metrics"], "comparison": result["comparison"], "regime": result["regime"], "elapsed_seconds": elapsed}, indent=2))
    elif args.command == "replay":
        from swing_trader.small_cap.settings_config import SmallCapSettings
        from .shadow import scan
        saved, inputs = read(args.scan), read(args.inputs)
        manifest = read(args.scan.parent / "manifest.json")
        if manifest.get("complete") is not True:
            raise ValueError("INCOMPLETE_SCAN_SNAPSHOT")
        for name, expected in manifest["sha256"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("INVALID_MANIFEST_PATH")
            if expected != digest((args.scan.parent / name).read_bytes()):
                raise ValueError("SCAN_SNAPSHOT_CHECKSUM_MISMATCH")
        if manifest["sha256"].get("inputs.json") != digest(args.inputs.read_bytes()):
            raise ValueError("REPLAY_INPUTS_CHECKSUM_MISMATCH")
        result = scan(inputs["snapshot"], inputs["preparation"], args.cache, SmallCapSettings.model_validate(inputs["settings"]),
                      inputs["decision_at"], inputs["legacy"], inputs["facts"])
        if result != saved:
            raise ValueError("REPLAY_MISMATCH_INPUT_OR_CODE_CHANGED")
        print(json.dumps({"identical": True, "scan_id": result["scan_id"], "result_sha256": object_id(result)}))
    elif args.command == "track":
        from .opportunity import register
        result = register(read(args.scan), args.cohort, read(args.previous) if args.previous else None)
        write_json(args.output, result)
        print(json.dumps({"observations": len(result["observations"]), "cohort": args.cohort}))
    elif args.command == "facts":
        import yfinance as yf
        from .facts import collect
        yf.set_tz_cache_location(str(args.cache / "yf_internal"))
        result = collect(read(args.scan), load_snapshot(args.snapshot), args.limit)
        write_json(args.output, result)
        print(json.dumps({"tickers": len(result), "known_market_cap": sum(f["market_cap"] is not None for f in result.values()),
                          "known_earnings": sum(f["earnings_coverage"] == "VERIFIED" for f in result.values())}))
    else:
        from .opportunity import forward
        result = forward(read(args.opportunities), read(args.preparation), args.cache)
        write_json(args.output, result)
        print(json.dumps({"observations": len(result["results"]), "status_counts": result["status_counts"]}))


if __name__ == "__main__":
    main()
