"""
Finviz ticker-parse regression tests (2026-07-17).

Finviz added a logo cell to the screener table: the ticker <td> now contains a
first-letter fallback span PLUS the ticker link. finvizfinance's ``td.text``
concatenates both text nodes (ARVN -> AARVN), which corrupted 143/146 universe
tickers and silently reduced a scan to 3 stocks. These tests lock in the
tab-link-based extraction in ``_ticker_safe_overview_cls``.
"""

import pandas as pd
from bs4 import BeautifulSoup

from swing_trader.small_cap.universe import _ticker_safe_overview_cls

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
    cls = _ticker_safe_overview_cls()
    assert cls._extract_ticker(_td(NEW_LAYOUT_TD)) == "ADNT"


def test_extract_ticker_old_layout():
    cls = _ticker_safe_overview_cls()
    assert cls._extract_ticker(_td(OLD_LAYOUT_TD)) == "ADNT"


def test_extract_ticker_boxover_fallback():
    cls = _ticker_safe_overview_cls()
    td = _td('<td data-boxover-ticker="GRPN"><span>G</span>GRPN-junk</td>')
    assert cls._extract_ticker(td) == "GRPN"


def test_extract_ticker_plain_text_last_resort():
    cls = _ticker_safe_overview_cls()
    assert cls._extract_ticker(_td("<td> XYZ </td>")) == "XYZ"


def test_get_table_parses_clean_tickers_and_keeps_other_columns():
    cls = _ticker_safe_overview_cls()
    soup = BeautifulSoup(NEW_LAYOUT_TABLE, "html.parser")
    rows = soup.find("table", class_="screener_table").find_all("tr")
    table_header = ["Ticker", "Company", "Price"]

    instance = cls.__new__(cls)  # skip __init__ (no network state needed)
    df = instance._get_table(rows, pd.DataFrame(), [2], table_header)

    assert list(df["Ticker"]) == ["ADNT", "GRPN"]
    assert list(df["Company"]) == ["Adient plc", "Groupon Inc"]
    assert [float(p) for p in df["Price"]] == [20.07, 30.10]
