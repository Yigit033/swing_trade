"""Offline deterministic tests; synthetic directories, no network or DB."""
import json

import pandas as pd
import pytest

from .master import build_snapshot, classify, load_snapshot, parse_directory, save_snapshot
from .comparison import compare, inspect_cache
from .legacy import summarize_legacy

STAMP = "2026-10-10T18:00:00+00:00"


def payloads():
    return {
        "nasdaqlisted": (
            "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
            "AAA|Alpha Common Stock|Q|N|N|100|N|N\n"
            "AAAW|Alpha Warrant|S|N|N|100|N|N\n"
            "QQQ|Index Trust|G|N|N|100|Y|N\n"
            "MYST|Unclear Issuer|S|N|N|100|N|N\n"
            "TEST|Test Common Stock|Q|Y|N|100|N|N\n"
            "File Creation Time: 1009202621:31|||||||\n").encode(),
        "otherlisted": (
            "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
            "BBB.A|Beta Class A Common Stock|N|BBB.A|N|100|N|BBB.A\n"
            "ADR|Foreign American Depositary Shares|N|ADR|N|100|N|ADR\n"
            "ARC|Arca Common Stock|P|ARC|N|100|N|ARC\n"
            "AMER|American Common Stock|A|AMER|N|100|N|AMER\n"
            "File Creation Time: 1009202621:31|||||||\n").encode(),
    }


@pytest.mark.parametrize("name,etf,expected", [
    ("Acquisition Units, each with Common Stock and Warrant", "N", "UNIT"),
    ("Corp Warrants to purchase Common Stock", "N", "WARRANT"),
    ("American Depositary Shares representing Preferred Shares", "N", "PREFERRED"),
    ("Foreign American Depositary Shares", "N", "ADR"),
    ("Issuer Class B Ordinary Shares", "N", "COMMON_STOCK"),
    ("Issuer 5% Notes", "N", "DEBT"),
    ("Issuer Rights", "N", "RIGHT"),
    ("Issuer - Right", "N", "RIGHT"),
    ("Issuer Common Subordinate Voting Shares", "N", "COMMON_STOCK"),
    ("Issuer American Depositary Shares representing the right to receive ordinary shares", "N", "ADR"),
    ("Index exchange traded notes", "Y", "ETN"),
    ("Unclear Trust", "Y", "ETF"),
    ("Income Fund Common Shares", "N", "FUND"),
    ("Mystery Corp", "N", "UNKNOWN"),
])
def test_security_precedence(name, etf, expected):
    assert classify(name, etf)["security_type"] == expected


def test_scope_unknowns_alias_and_not_pit():
    snapshot = build_snapshot(payloads(), STAMP)
    rows = {r["symbol"]: r for r in snapshot["records"]}
    assert rows["ARC"]["discovery_status"] == "OUT_OF_SCOPE"
    assert rows["AMER"]["discovery_status"] == "OUT_OF_SCOPE"
    assert rows["TEST"]["discovery_reason"] == "official_test_issue"
    assert rows["MYST"]["discovery_status"] == "REVIEW_REQUIRED"
    assert rows["BBB.A"]["yfinance_symbol_candidate"] == "BBB-A"
    assert rows["AAA"]["market_cap"]["status"] == "UNKNOWN"
    assert rows["AAA"]["ipo_status"] == "UNKNOWN"
    assert rows["AAA"]["spac_status"] == "UNKNOWN"
    assert snapshot["historical_point_in_time"] is False
    assert snapshot["summary"]["by_exchange"] == {"NASDAQ": 5, "NYSE": 2}


def test_spac_is_suspicion_not_fact():
    assert classify("Alpha Acquisition Corp Common Stock", "N")["spac_status"] == "SUSPECTED_NAME"


def test_snapshot_replay_integrity_and_no_overwrite(tmp_path):
    raw = payloads()
    snapshot = build_snapshot(raw, STAMP)
    path = save_snapshot(tmp_path, snapshot, raw)
    assert load_snapshot(path) == snapshot
    with pytest.raises(FileExistsError):
        save_snapshot(tmp_path, snapshot, raw)
    (path / "nasdaqlisted.txt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_snapshot(path)


def test_both_sources_and_timezone_required():
    with pytest.raises(ValueError, match="Both"):
        build_snapshot({"nasdaqlisted": payloads()["nasdaqlisted"]}, STAMP)
    with pytest.raises(ValueError, match="timezone"):
        build_snapshot(payloads(), "2026-10-10T18:00:00")


@pytest.mark.parametrize("mutation", ["footer", "schema", "duplicate", "flag", "truncated"])
def test_directory_fails_closed_on_corruption(mutation):
    raw = payloads()["nasdaqlisted"]
    if mutation == "footer":
        raw = raw.split(b"File Creation")[0]
    elif mutation == "schema":
        raw = raw.replace(b"Test Issue", b"Test Unknown")
    elif mutation == "duplicate":
        raw = raw.replace(b"AAAW|", b"AAA|")
    elif mutation == "flag":
        raw = raw.replace(b"|Q|N|", b"|Q|X|")
    else:
        raw = raw.replace(b"AAA|Alpha Common Stock|Q|N|N|100|N|N", b"AAA|Alpha")
    with pytest.raises(ValueError):
        parse_directory(raw, "nasdaqlisted", STAMP)


def fixture_legacy():
    from swing_trader.small_cap.settings_config import SmallCapSettings
    settings = SmallCapSettings().universe_scan.model_dump()
    settings.update(max_scan_tickers=1, min_finviz_tickers_skip_static_merge=1)
    queries = [{"status": "OK", "rows": rows} for rows in [
        [{"Ticker": "AAA", "Price": 10, "Volume": 100}, {"Ticker": "BBB.A", "Price": 20, "Volume": 200}],
        [{"Ticker": "AAA", "Price": 10, "Volume": 100}],
        [{"Ticker": "BAD", "Price": None, "Volume": 100}, {"Ticker": "CHEAP", "Price": 2, "Volume": 100}],
        [{"Ticker": "BAN", "Price": 20, "Volume": 100}],
    ]]
    return summarize_legacy(queries, settings, ["BAN", "ADR"], ["BBB.A"])


def test_funnel_attributes_sequential_losses_and_counts_overlap():
    result = fixture_legacy()
    assert result["complete"] is True
    assert result["funnel"] == {
        "query_rows_with_overlap": 6, "unique_query_members": 5,
        "duplicate_query_rows": 1, "excluded": ["BAN"], "missing_price": ["BAD"],
        "outside_price": ["CHEAP"], "after_price": 2, "cap_cut": ["AAA"],
        "finviz_selected_count": 1, "final_selected_count": 1,
    }
    assert result["upstream_per_predicate_rejection_counts"] is None


def test_partial_legacy_not_reported_complete():
    from swing_trader.small_cap.settings_config import SmallCapSettings
    result = summarize_legacy([{"status": "FAILED", "rows": []}], SmallCapSettings().universe_scan.model_dump(), [], ["AAA"])
    assert result["complete"] is False
    assert result["static_added"] == ["AAA"]


def test_comparison_separates_potential_exclusion_from_observed_loss():
    result = compare(build_snapshot(payloads(), STAMP), fixture_legacy())
    assert result["candidate_not_in_legacy"] == ["AAA", "ADR"]
    assert result["listed_hardcoded_exclusions"][0]["observed_finviz_rejection"] is False
    assert result["legacy_finviz_count"] == 1
    assert result["same_market_session_verified"] is False


def test_class_alias_is_matched_without_collapsing_share_classes():
    legacy = fixture_legacy()
    legacy["selected"] = ["BBB-A"]
    result = compare(build_snapshot(payloads(), STAMP), legacy)
    assert result["legacy_in_scope_count"] == 1
    assert result["candidate_not_in_legacy"] == ["AAA", "ADR"]
    assert result["symbol_resolution"]["BBB-A"]["status"] == "ALIAS_MATCH"


def test_failed_finviz_is_unknown_not_zero_coverage():
    legacy = fixture_legacy()
    legacy["complete"] = False
    assert compare(build_snapshot(payloads(), STAMP), legacy)["legacy_finviz_count"] is None


def test_recorded_run_requires_complete_membership():
    from .legacy import recorded_run_legacy
    run = {"id": 1, "created_at": STAMP, "stats": {"stocks_scanned": 2,
           "scanned_members": [{"ticker": "AAA"}], "universe_ranked_total": 2}}
    result = recorded_run_legacy(run)
    assert result["complete"] is False
    assert result["funnel"]["excluded"] is None


def test_snapshot_incomplete_manifest_is_rejected(tmp_path):
    raw = payloads()
    path = save_snapshot(tmp_path, build_snapshot(raw, STAMP), raw)
    (path / "manifest.json").write_text(json.dumps({"complete": True, "sha256": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete"):
        load_snapshot(path)


def test_collector_executes_actual_champion_queries_and_cap(monkeypatch):
    import swing_trader.small_cap.universe as module
    from .legacy import collect_legacy
    calls = []
    class FakeOverview:
        def set_filter(self, filters_dict):
            calls.append(filters_dict)
        def screener_view(self):
            return pd.DataFrame([{"Ticker": "AAA", "Price": 10, "Volume": 1e6}])
    monkeypatch.setattr(module, "_ticker_safe_overview_cls", lambda: FakeOverview)
    result = collect_legacy({"universe_scan": {"min_finviz_tickers_skip_static_merge": 1}})
    assert len(calls) == 4
    assert sum(c.get("Relative Volume") == "Over 2" for c in calls) == 2
    assert sum(c.get("20-Day High/Low") == "New High" for c in calls) == 2
    assert result["selected"] == result["finviz_selected"] == ["AAA"]


def test_cache_only_past_bars_stale_missing_and_invalid(monkeypatch):
    from swing_trader.small_cap.settings_config import SmallCapSettings
    from swing_trader.small_cap.signals import SmallCapSignals
    seen = []
    def observe(self, df):
        seen.append(df["Date"].max())
        return False, "fixture", {}
    monkeypatch.setattr(SmallCapSignals, "check_vce_breakout", observe)
    monkeypatch.setattr(SmallCapSignals, "check_rvol_thrust", observe)
    dates = pd.bdate_range("2026-01-01", periods=80)
    df = pd.DataFrame({"Date": dates, "Open": 10., "High": 11., "Low": 9., "Close": 10., "Volume": 1e6})
    cutoff = dates[59].strftime("%Y-%m-%d")
    result = inspect_cache(build_snapshot(payloads(), STAMP), fixture_legacy(),
                           {"AAA": df, "BBB-A": df.iloc[:50]}, SmallCapSettings(), cutoff, {})
    assert seen and all(date <= dates[59] for date in seen)
    assert result["coverage"]["trigger_evaluable"] == 1
    assert result["coverage"]["stale_for_as_of"] == 1
    assert result["coverage"]["no_cached_symbol"] == 1
    assert result["groups"]["all_candidates"]["denominator"] == 3


def test_master_is_not_capped_at_300():
    raw = payloads()
    header, *_, footer = raw["nasdaqlisted"].decode().splitlines()
    raw["nasdaqlisted"] = (header + "\n" + "\n".join(f"S{i}|Issuer Common Stock|Q|N|N|100|N|N" for i in range(350)) + "\n" + footer).encode()
    assert build_snapshot(raw, STAMP)["summary"]["by_exchange"]["NASDAQ"] == 350
