#!/usr/bin/env python3
"""
Compare two SmallCap backtest API JSON exports (saved response bodies).

Usage:
  python scripts/compare_backtest_json.py run_a.json run_b.json

Expects top-level keys like metrics, diagnostics, trades, exit_stats (inside metrics).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def load(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("error"):
        raise SystemExit(f"{path}: API error — {data.get('error')}")
    if "metrics" not in data and "results" in data:
        inner = data["results"]
        if inner is None:
            raise SystemExit(f"{path}: results is null")
        data = inner
    return data


def fmt_num(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def diff_metrics(a: Dict, b: Dict, label_a: str, label_b: str) -> None:
    keys = [
        "total_return",
        "total_pnl_dollar",
        "win_rate",
        "profit_factor",
        "max_drawdown",
        "total_trades",
        "winning_trades",
        "losing_trades",
        "avg_hold_days",
    ]
    print("=== metrics ===")
    for k in keys:
        va = a.get(k)
        vb = b.get(k)
        if va is None and vb is None:
            continue
        print(f"  {k}:")
        print(f"    {label_a}: {fmt_num(va)}")
        print(f"    {label_b}: {fmt_num(vb)}")
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            print(f"    delta (B-A): {fmt_num(vb - va)}")


def diff_flat_dict(a: Optional[Dict], b: Optional[Dict], title: str) -> None:
    print(f"=== {title} ===")
    if not a and not b:
        print("  (yok)")
        return
    keys = sorted(set((a or {}).keys()) | set((b or {}).keys()))
    for k in keys:
        va = (a or {}).get(k)
        vb = (b or {}).get(k)
        if va == vb:
            print(f"  {k}: {fmt_num(va)}")
        else:
            print(f"  {k}: A={fmt_num(va)}  B={fmt_num(vb)}  (B-A)={fmt_num((vb or 0) - (va or 0)) if isinstance(va,(int,float)) and isinstance(vb,(int,float)) else '—'}")


def diff_exit_stats(a: Dict, b: Dict, label_a: str, label_b: str) -> None:
    ea = a.get("exit_stats") or {}
    eb = b.get("exit_stats") or {}
    if not ea and not eb:
        return
    print("=== exit_stats (metrics içi) ===")
    keys = sorted(set(ea.keys()) | set(eb.keys()))
    for k in keys:
        ca = ea.get(k, {})
        cb = eb.get(k, {})
        na = ca.get("count") if isinstance(ca, dict) else None
        nb = cb.get("count") if isinstance(cb, dict) else None
        print(f"  {k}: {label_a} count={na}  {label_b} count={nb}")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two backtest JSON files.")
    p.add_argument("file_a", type=Path, help="Baseline JSON")
    p.add_argument("file_b", type=Path, help="New run JSON")
    p.add_argument("--label-a", default="A (baseline)", help="Label for first file")
    p.add_argument("--label-b", default="B (new)", help="Label for second file")
    args = p.parse_args()

    da = load(args.file_a)
    db = load(args.file_b)

    ma = da.get("metrics") or {}
    mb = db.get("metrics") or {}

    print(f"Comparing:\n  {args.label_a}: {args.file_a}\n  {args.label_b}: {args.file_b}\n")

    diff_metrics(ma, mb, args.label_a, args.label_b)
    print()
    diff_flat_dict(da.get("diagnostics"), db.get("diagnostics"), "diagnostics (top-level)")
    print()
    diff_exit_stats(ma, mb, args.label_a, args.label_b)
    print()
    ta = len(da.get("trades") or [])
    tb = len(db.get("trades") or [])
    print(f"=== trades length ===\n  {args.label_a}: {ta}\n  {args.label_b}: {tb}\n")


if __name__ == "__main__":
    main()
