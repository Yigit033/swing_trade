"""Official current symbol directories, conservative classification, immutable snapshots.

A retrieval is evidence of membership at observation time ONLY. Neither file
contains IPO dates, historical membership, market cap, float or fundamentals.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

SOURCES = {
    "nasdaqlisted": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "otherlisted": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
}
EXCHANGES = {"N": "NYSE", "A": "NYSE_AMERICAN", "P": "NYSE_ARCA",
             "Z": "CBOE_BZX", "V": "IEX"}
SCHEMA_VERSION = 1
CLASSIFIER_VERSION = "phase1-v3"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Exclusive create: never silently overwrite audit artifacts."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def classify(name: str, etf: str, nextshares: str = "N") -> dict:
    """Type from official flags + name evidence; heuristics are explicitly tagged.

    Do not infer warrant/preferred from ticker suffix alone. Unit descriptions
    often mention warrants and common shares, so unit takes precedence.
    """
    rules = [
        ("ETN", r"\bETNs?\b|exchange.traded notes?"),
        ("UNIT", r"\bunits?\b"),
        ("WARRANT", r"\bwarrants?\b"),
        ("PREFERRED", r"\bpreferred\b|\bpreference\b"),
        ("RIGHT", r"\brights\b|\bright\s*$"),
        ("DEBT", r"\bnotes\b|\bdebentures\b|\bbonds\b"),
        ("FUND", r"\bfund\b|\bportfolio\b|\bETF\b"),
        ("ADR", r"american deposit[ao]ry|\bADRs?\b|\bADSs?\b"),
        ("COMMON_STOCK", r"\bcommon (?:subordinate voting )?(?:stock|shares?)\b|\b(?:ordinary|ord) shares?\b|\bcapital stock\b"),
    ]
    if re.search(r"\bETNs?\b|exchange.traded notes?", name, re.I):
        kind, evidence = "ETN", "name:ETN"
    elif etf == "Y":
        kind, evidence = "ETF", "official:ETF=Y"
    elif nextshares == "Y":
        kind, evidence = "FUND", "official:NextShares=Y"
    else:
        kind, evidence = "UNKNOWN", "no_unambiguous_type_in_directory"
        for candidate, pattern in rules:
            if re.search(pattern, name, re.I):
                kind, evidence = candidate, f"name:{candidate}"
                break
    share_class = re.search(r"\bclass\s+([A-Z0-9]+)\b", name, re.I)
    return {
        "security_type": kind,
        "classification_basis": evidence,
        "classification_confidence": "OFFICIAL_FLAG" if evidence.startswith("official:") else
            ("UNKNOWN" if kind == "UNKNOWN" else "NAME_HEURISTIC"),
        "adr_status": "YES_NAME_EVIDENCE" if kind == "ADR" else "UNKNOWN",
        "spac_status": "SUSPECTED_NAME" if re.search(r"\bacquisition\b|\bSPAC\b|blank.check", name, re.I) else "UNKNOWN",
        "share_class": share_class.group(1).upper() if share_class else None,
        "multiple_classes_status": "CLASS_DESIGNATED" if share_class else "UNKNOWN",
        "ipo_date": None,
        "ipo_status": "UNKNOWN",
    }


def parse_directory(raw: bytes, source: str, observed_at: str) -> tuple[list[dict], dict]:
    if source not in SOURCES:
        raise ValueError("Unsupported official directory")
    symbol_field = "Symbol" if source == "nasdaqlisted" else "ACT Symbol"
    lines = [line for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    footer = [line.split("|", 1)[0] for line in lines if line.startswith("File Creation Time:")]
    if len(footer) != 1 or not lines[-1].startswith("File Creation Time:"):
        raise ValueError("Missing or misplaced directory footer; possible partial download")
    reader = csv.DictReader(io.StringIO("\n".join(lines[:-1])), delimiter="|")
    required = {symbol_field, "Security Name", "Test Issue", "ETF"}
    if source == "otherlisted":
        required |= {"Exchange", "CQS Symbol", "NASDAQ Symbol"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("Unexpected directory schema")
    records, seen = [], set()
    for row in reader:
        if None in row or any(row.get(key) is None for key in required):
            raise ValueError("Malformed directory row")
        symbol = row[symbol_field].strip()
        if not symbol or symbol in seen or not row["Security Name"].strip():
            raise ValueError("Empty or duplicate directory symbol")
        seen.add(symbol)
        if row["Test Issue"] not in {"N", "Y"} or row["ETF"] not in {"N", "Y"}:
            raise ValueError("Unknown official flag value")
        exchange = "NASDAQ" if source == "nasdaqlisted" else EXCHANGES.get(row["Exchange"], "UNKNOWN")
        info = classify(row["Security Name"], row["ETF"], row.get("NextShares", "N"))
        in_scope = exchange in {"NASDAQ", "NYSE"}
        test = row["Test Issue"] == "Y"
        kind = info["security_type"]
        if not in_scope:
            status, reason = "OUT_OF_SCOPE", "exchange_outside_NASDAQ_NYSE"
        elif test:
            status, reason = "EXCLUDED", "official_test_issue"
        elif kind in {"COMMON_STOCK", "ADR"}:
            status, reason = "CANDIDATE", "equity_candidate_cap_liquidity_not_yet_evaluated"
        elif kind == "UNKNOWN":
            status, reason = "REVIEW_REQUIRED", "security_type_unknown_retained"
        else:
            status, reason = "EXCLUDED", f"security_type_{kind}"
        # Safe class-share spelling candidate, NOT a verified vendor identifier.
        if re.fullmatch(r"[A-Z]+", symbol):
            yf_symbol, mapping = symbol, "EXACT_SYMBOL_UNVERIFIED"
        elif kind == "COMMON_STOCK" and info["share_class"] and re.fullmatch(r"[A-Z]+\.[A-Z]", symbol):
            yf_symbol, mapping = symbol.replace(".", "-"), "CLASS_ALIAS_UNVERIFIED"
        else:
            yf_symbol, mapping = None, "REVIEW_REQUIRED"
        unknown = lambda: {"value": None, "status": "UNKNOWN", "source": None, "observed_at": None}
        records.append({
            "id": f"{source}:{symbol}", "symbol": symbol, "exchange": exchange,
            "source": source, "source_exchange_code": row.get("Exchange", "Q"),
            "observed_at": observed_at, "test_issue": test, **info,
            "discovery_status": status, "discovery_reason": reason,
            "yfinance_symbol_candidate": yf_symbol, "symbol_mapping_status": mapping,
            "market_cap": unknown(), "float_shares": unknown(), "fundamentals_status": "UNKNOWN",
            "cap_band": "UNKNOWN", "first_seen_at": observed_at,
            "raw": row,
        })
    if not records:
        raise ValueError("Empty directory")
    return records, {"source": source, "url": SOURCES[source], "sha256": digest(raw),
                     "bytes": len(raw), "rows": len(records), "observed_at": observed_at,
                     "file_creation_time_raw": footer[0],
                     "file_creation_timezone": "UNSPECIFIED_BY_FILE"}


def universe_summary(records: list[dict]) -> dict:
    scope = [r for r in records if r["exchange"] in {"NASDAQ", "NYSE"}]
    return {"all_source_rows": len(records), "scope_rows_including_test": len(scope),
            "scope_non_test": sum(not r["test_issue"] for r in scope),
            "by_exchange": dict(sorted(Counter(r["exchange"] for r in scope).items())),
            "security_types": dict(sorted(Counter(r["security_type"] for r in scope).items())),
            "discovery_status": dict(sorted(Counter(r["discovery_status"] for r in scope).items())),
            "by_exchange_security_type": {e: dict(sorted(Counter(r["security_type"] for r in scope if r["exchange"] == e).items())) for e in ("NASDAQ", "NYSE")},
            "spac_suspected": sum(r["spac_status"] == "SUSPECTED_NAME" for r in scope),
            "class_designated": sum(r["share_class"] is not None for r in scope),
            "unknown_market_cap": sum(r["market_cap"]["status"] == "UNKNOWN" for r in scope),
            "unknown_float": sum(r["float_shares"]["status"] == "UNKNOWN" for r in scope)}


def build_snapshot(payloads: dict[str, bytes], observed_at: str, headers: dict | None = None) -> dict:
    parsed_time = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    if parsed_time.utcoffset() is None:
        raise ValueError("Observation timestamp must include timezone")
    if set(payloads) != set(SOURCES):
        raise ValueError("Both official sources are required; partial universe not published")
    records, metadata = [], []
    for source in sorted(SOURCES):
        rows, meta = parse_directory(payloads[source], source, observed_at)
        meta["http_headers"] = (headers or {}).get(source, {})
        metadata.append(meta)
        records.extend(rows)
    records.sort(key=lambda r: (r["exchange"], r["symbol"], r["source"]))
    return {"schema_version": SCHEMA_VERSION, "classifier_version": CLASSIFIER_VERSION,
            "observed_at": observed_at, "historical_point_in_time": False,
            "membership_semantics": "CURRENT_OBSERVATION_ONLY_NOT_VALID_FOR_EARLIER_DATES",
            "sources": metadata, "summary": universe_summary(records), "records": records}


def fetch_snapshot() -> tuple[dict, dict[str, bytes]]:
    """Two public HTTPS downloads, bounded size/time; no keys or proxy rotation."""
    payloads, headers = {}, {}
    for name, url in SOURCES.items():
        request = Request(url, headers={"User-Agent": "SwingDiscoveryResearch/1.0"})
        with urlopen(request, timeout=30) as response:
            payload = response.read(10_000_001)
            if len(payload) > 10_000_000:
                raise ValueError("Directory exceeds size limit")
            payloads[name] = payload
            headers[name] = {key: response.headers.get(key) for key in
                             ("Date", "Last-Modified", "ETag", "Content-Type")}
    return build_snapshot(payloads, utc_now(), headers), payloads


def save_snapshot(root: Path, snapshot: dict, payloads: dict[str, bytes]) -> Path:
    # Unique exclusive directory + completion manifest last. Partial writes are
    # never accepted by load_snapshot. Raw downloads allow exact offline replay.
    stamp = datetime.fromisoformat(snapshot["observed_at"]).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = root / stamp
    path.mkdir(parents=True, exist_ok=False)
    for source in SOURCES:
        with (path / f"{source}.txt").open("xb") as handle:
            handle.write(payloads[source])
    write_json(path / "universe.json", snapshot)
    files = ["universe.json", *(f"{s}.txt" for s in SOURCES)]
    write_json(path / "manifest.json", {"complete": True, "sha256": {f: digest((path / f).read_bytes()) for f in files}})
    return path


def load_snapshot(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    expected = {"universe.json", *(f"{s}.txt" for s in SOURCES)}
    if manifest.get("complete") is not True or set(manifest.get("sha256", {})) != expected:
        raise ValueError("Incomplete snapshot manifest")
    for name, checksum in manifest["sha256"].items():
        if digest((path / name).read_bytes()) != checksum:
            raise ValueError("Snapshot checksum mismatch")
    snapshot = json.loads((path / "universe.json").read_text(encoding="utf-8"))
    rebuilt = build_snapshot({s: (path / f"{s}.txt").read_bytes() for s in SOURCES}, snapshot["observed_at"],
                             {m["source"]: m["http_headers"] for m in snapshot["sources"]})
    if rebuilt != snapshot:
        raise ValueError("Snapshot cannot be reproduced with this schema/classifier")
    return snapshot
