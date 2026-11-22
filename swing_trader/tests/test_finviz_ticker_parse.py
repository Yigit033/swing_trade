"""
Finviz ticker-parse regression tests (2026-07-17).

Finviz added a logo cell to the screener table: the ticker <td> now contains a
first-letter fallback span PLUS the ticker link. finvizfinance's ``td.text``
concatenates both text nodes (ARVN -> AARVN), which corrupted 143/146 universe
tickers and silently reduced a scan to 3 stocks. These tests lock in the
tab-link-based extraction for OLD finvizfinance (``_legacy_ticker_safe_overview_cls``).

2026-11-22: finvizfinance 1.7 fixed tickers upstream (cell data attributes) and changed
``_get_table`` to take cell lists; our legacy override then broke EVERY query and the scan
silently fell back to the static list. ``_ticker_safe_overview_cls`` now picks by version and
a total Finviz failure is surfaced (``static_fallback`` → scanner stops).
"""

import pandas as pd
from bs4 import BeautifulSoup

from swing_trader.small_cap import universe as U
from swing_trader.small_cap.universe import _legacy_ticker_safe_overview_cls

# Mirrors the live Finviz HTML captured on 2026-07-17 (tight markup, no
# whitespace between nodes — exactly how td.text produced "AADNT").
NEW_LAYOUT_TD = (
    '<td data-boxover-ticker="ADNT" align="left">'
    '<span class="flex items-center gap-1 pl-0.5">'
    '<a class="company-ticker" href="stock?t=ADNT"><img src="x.svg"/><span>A</span></a>'
    '<a class="tab-link" href="stock?t=ADNT">ADNT</a>'
    "</span></td>"
)

OLD_LAYOUT_TD = '<td><a class="tab-link">ADNT</a></td>'

NEW_LAYOUT_TABLE = """
<table class="screener_table">
  <tr><th>No.</th><th>Ticker</th><th>Company</th><th>Price</th></tr>
  <tr>
    <td>1</td>
    <td data-boxover-ticker="ADNT"><span><a class="company-ticker"><span>A</span></a><a class="tab-link">ADNT</a></span></td>
    <td><a class="tab-link">Adient plc</a></td>
    <td><a class="tab-link">20.07</a></td>
  </tr>
  <tr>
    <td>2</td>
    <td data-boxover-ticker="GRPN"><span><a class="company-ticker"><span>G</span></a><a class="tab-link">GRPN</a></span></td>
    <td><a class="tab-link">Groupon Inc</a></td>
    <td><a class="tab-link">30.10</a></td>
  </tr>
</table>
"""


def _td(html: str):
    return BeautifulSoup(html, "html.parser").find("td")


def test_raw_td_text_still_doubles_first_letter():
    # Documents the upstream bug: if this stops failing on .text, Finviz
    # reverted their layout and the patch is a no-op (still safe).
    assert _td(NEW_LAYOUT_TD).text.strip() == "AADNT"


def test_extract_ticker_new_logo_layout():
    cls = _legacy_ticker_safe_overview_cls()
    assert cls._extract_ticker(_td(NEW_LAYOUT_TD)) == "ADNT"


def test_extract_ticker_old_layout():
    cls = _legacy_ticker_safe_overview_cls()
    assert cls._extract_ticker(_td(OLD_LAYOUT_TD)) == "ADNT"


def test_extract_ticker_boxover_fallback():
    cls = _legacy_ticker_safe_overview_cls()
    td = _td('<td data-boxover-ticker="GRPN"><span>G</span>GRPN-junk</td>')
    assert cls._extract_ticker(td) == "GRPN"


def test_extract_ticker_plain_text_last_resort():
    cls = _legacy_ticker_safe_overview_cls()
    assert cls._extract_ticker(_td("<td> XYZ </td>")) == "XYZ"


def test_get_table_parses_clean_tickers_and_keeps_other_columns():
    cls = _legacy_ticker_safe_overview_cls()
    soup = BeautifulSoup(NEW_LAYOUT_TABLE, "html.parser")
    rows = soup.find("table", class_="screener_table").find_all("tr")
    table_header = ["Ticker", "Company", "Price"]

    instance = cls.__new__(cls)  # skip __init__ (no network state needed)
    df = instance._get_table(rows, pd.DataFrame(), [2], table_header)

    assert list(df["Ticker"]) == ["ADNT", "GRPN"]
    assert list(df["Company"]) == ["Adient plc", "Groupon Inc"]
    assert [float(p) for p in df["Price"]] == [20.07, 30.10]


# ── 2026-11-22: sürüme göre sınıf seçimi + çöküşün görünür olması ────────────

def test_new_library_uses_upstream_parser(monkeypatch):
    from finvizfinance.screener.overview import Overview
    monkeypatch.setattr(U, "_TICKER_SAFE_OVERVIEW_CLS", None)
    monkeypatch.setattr(U, "_upstream_parses_tickers", lambda: True)
    assert U._ticker_safe_overview_cls() is Overview          # yeni kütüphane ezilmez
    monkeypatch.setattr(U, "_TICKER_SAFE_OVERVIEW_CLS", None)
    monkeypatch.setattr(U, "_upstream_parses_tickers", lambda: False)
    assert U._ticker_safe_overview_cls() is _legacy_ticker_safe_overview_cls()
    monkeypatch.setattr(U, "_TICKER_SAFE_OVERVIEW_CLS", None)


def test_installed_library_is_detected_as_upstream_fixed():
    """requirements.txt finvizfinance==1.7.1 — kurulu sürüm yeni ayrıştırıcıyı taşımalı."""
    assert U._upstream_parses_tickers() is True


class _FakeOverview:
    frames = []

    def set_filter(self, filters_dict):
        pass

    def screener_view(self, verbose=0):
        r = _FakeOverview.frames.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _provider(monkeypatch, frames):
    from swing_trader.small_cap.settings_config import SmallCapSettings
    monkeypatch.setattr(U, "_ticker_safe_overview_cls", lambda: _FakeOverview)
    _FakeOverview.frames = list(frames)
    return U.SmallCapUniverse(settings=SmallCapSettings())


def _rows(*tickers):
    return pd.DataFrame({"Ticker": list(tickers), "Price": [20.0] * len(tickers),
                         "Volume": ["1.0M"] * len(tickers), "Market Cap": ["1.0B"] * len(tickers)})


def test_all_queries_failing_is_static_fallback_not_silent(monkeypatch):
    prov = _provider(monkeypatch, [AttributeError("'list' object has no attribute 'find_all'")] * 4)
    tickers = prov.get_universe(use_finviz=True)
    assert prov.last_universe_source == "static_fallback" and len(prov._last_query_errors) == 4
    assert tickers                                             # liste döner, tarayıcı durdurur


def test_quiet_market_merge_is_not_a_failure(monkeypatch):
    prov = _provider(monkeypatch, [_rows("ENR", "HZO"), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()])
    tickers = prov.get_universe(use_finviz=True)
    assert prov.last_universe_source == "finviz+static" and tickers[:2] == ["ENR", "HZO"]


def test_invalid_tickers_dropped_and_all_invalid_counts_as_error(monkeypatch):
    prov = _provider(monkeypatch, [_rows("ADNT", "AADNT-junk!")])
    df = prov._run_finviz_query({}, "Q")
    assert list(df["Ticker"]) == ["ADNT"] and prov._last_query_errors == []
    prov = _provider(monkeypatch, [_rows("bad ticker", "??")])
    assert prov._run_finviz_query({}, "Q").empty and len(prov._last_query_errors) == 1


def test_all_queries_silently_empty_is_also_a_failure(monkeypatch):
    """finvizfinance 1.5 yeni Finviz sayfasında hata vermeden boş dönüyordu (Kasım 2026)."""
    prov = _provider(monkeypatch, [pd.DataFrame()] * 4)
    prov.get_universe(use_finviz=True)
    assert prov.last_universe_source == "static_fallback" and prov._last_query_errors == []
