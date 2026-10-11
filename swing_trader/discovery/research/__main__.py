"""Offline CLI; no network, DB, scheduler, or production writes."""
import argparse
import json
from pathlib import Path
from collections import Counter
from datetime import date
from ..bars import read_object, object_id
from ..master import write_json, utc_now
from ..calendar import sessions
from .protocol import PROTOCOL


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inventory(preparation, cache):
    counts, starts, ends = Counter(), Counter(), Counter()
    coverage = Counter()
    refs = {}
    for outcome in preparation["outcomes"]:
        if not outcome.get("object_id"):
            counts[outcome["status"]] += 1
            continue
        obj = read_object(cache, outcome["object_id"])
        rows = obj["rows"]
        if outcome["symbol"] in ("SPY", "^VIX"):
            refs[outcome["symbol"]] = {"first": rows[0]["Date"], "last": rows[-1]["Date"], "bars": len(rows)}
            continue
        counts["cache_objects"] += 1
        starts[rows[0]["Date"]] += 1
        ends[rows[-1]["Date"]] += 1
        for i, row in enumerate(rows):
            if i >= PROTOCOL["min_history"]-1:
                # Inventory only: candidate sessions, stricter continuity checked in analysis.
                coverage[row["Date"]] += 1
    calendar = sessions(date.fromisoformat(min(starts)), date.fromisoformat(preparation["session"]))
    mature = calendar[PROTOCOL["min_history"]-1:-30]
    chosen = mature[-90:]
    return {"created_at": utc_now(), "preparation_id": object_id(preparation), "protocol": PROTOCOL,
            "counts": dict(counts), "first_dates": dict(starts), "last_dates": dict(ends), "references": refs,
            "available_mature_sessions": len(mature), "chosen_sessions": [str(d) for d in chosen],
            "coverage_min_history_only": {str(d): coverage[str(d)] for d in chosen},
            "future_outcomes_not_inspected": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["inventory", "run", "summarize"])
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--inputs", type=Path, help="Phase2 frozen inputs; current facts NEVER used historically")
    p.add_argument("--inventory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    prep = read(args.preparation)
    if args.command == "inventory":
        result = inventory(prep, args.cache)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
        print(json.dumps({k:v for k,v in result.items() if k not in ("first_dates", "coverage_min_history_only", "chosen_sessions")}, indent=2))
        print(json.dumps({"first": result["chosen_sessions"][0], "last": result["chosen_sessions"][-1],
                          "n": len(result["chosen_sessions"]), "coverage_min": min(result["coverage_min_history_only"].values())}))
    elif args.command == "run":
        from .panel import run
        run(read(args.inputs), prep, args.cache, read(args.inventory), args.output)
    else:
        from .analysis import summarize
        summarize(args.output)


if __name__ == "__main__":
    main()
