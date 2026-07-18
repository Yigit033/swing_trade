"""Huni 3. aşama telemetrisi — build_rank_info birim testleri."""

import pandas as pd

from swing_trader.small_cap.universe import build_rank_info


def _ranked_df(tickers):
    # get_finviz_universe çağırmadan önce df composite_score'a göre sıralanmış olur
    return pd.DataFrame({"Ticker": tickers})


def test_rank_info_with_cap_cut():
    df = _ranked_df(["AAA", "BBB", "CCC", "DDD", "EEE"])
    info = build_rank_info(df, cap=3)
    assert info["ranked_total"] == 5
    assert info["cap"] == 3
    assert info["ranks"] == {"AAA": 1, "BBB": 2, "CCC": 3, "DDD": 4, "EEE": 5}
    assert info["cut_tickers"] == ["DDD", "EEE"]


def test_rank_info_no_cut_when_under_cap():
    df = _ranked_df(["AAA", "BBB"])
    info = build_rank_info(df, cap=260)
    assert info["cut_tickers"] == []
    assert info["ranks"]["BBB"] == 2


def test_rank_info_signal_lookup_semantics():
    # Scanner sinyal ticker'ının sırasını ranks.get ile okur; evrende olmayan
    # (ör. static fallback'ten gelen) ticker None döner — kod bunu tolere eder.
    info = build_rank_info(_ranked_df(["AAA"]), cap=1)
    assert info["ranks"].get("YOK") is None
