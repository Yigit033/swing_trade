"""
POST /api/scanner/track — sunucu tarafı pozisyon boyutu regresyon testleri (2026-08-03).

Canlı adli tıp: paper hesapta üç işlem aşırı boyutluydu.

    NSP  entry 48.89  stop 44.91  ->  2901 hisse = 141.830$ pozisyon / 11.546$ risk
                                     (motorun doğru cevabı: 37 hisse / 147$ risk)
    TH   entry 20.12  stop 19.12  ->   773 hisse (doğru: 124)      6x
    CPRX entry 31.38  stop 31.30  ->   796 hisse (doğru:  79)     10x
                                       (tam olarak portfolio_value=100.000$ ile eşleşiyor)

Motor (risk.py calculate_position_size) doğru hesaplıyordu; sızıntı endpoint'teydi:
`position_size` istemciden ne gelirse doğrudan DB'ye yazılıyordu. Sonuç: -1.92%'lik
bir fiyat hareketi -2.727$ zarara dönüştü ve hesabın toplamını (~başabaş → -3.343$)
tersine çevirdi — strateji değil, boyutlandırma hatası.

Kural: risk-kritik bir sayı asla istemciden alınmaz. İstemcinin değeri yalnızca
ÜST SINIR olarak kabul edilir (kullanıcı daha küçük pozisyon isteyebilir).
"""

import pytest

from api.routers.scanner import TrackSignalRequest, _authoritative_position_size


def _req(**kw):
    base = dict(ticker="TEST", entry_price=48.89, stop_loss=44.91,
                target_1=55.0, swing_type="A", portfolio_value=10000)
    base.update(kw)
    return TrackSignalRequest(**base)


# ── Canlı arızanın birebir tekrarı ────────────────────────────────────────

@pytest.mark.parametrize("ticker,entry,stop,client_size,expected_max", [
    ("NSP", 48.89, 44.91, 2901, 37),
    ("TH", 20.12, 19.12, 773, 124),
    ("CPRX", 31.38, 31.30, 796, 79),
])
def test_oversized_client_size_is_capped(ticker, entry, stop, client_size, expected_max):
    """İstemcinin şişmiş boyutu sunucu tavanına çekilmeli."""
    size = _authoritative_position_size(
        _req(ticker=ticker, entry_price=entry, stop_loss=stop, position_size=client_size)
    )
    assert size == expected_max, f"{ticker}: {client_size} -> {size}, beklenen {expected_max}"


def test_nsp_risk_stays_within_budget():
    """NSP senaryosunda risk 1.5% bütçesini (150$) aşmamalı — canlıda 11.546$'dı."""
    size = _authoritative_position_size(_req(position_size=2901))
    risk = size * (48.89 - 44.91)
    assert risk <= 10000 * 0.015 + 4.0, f"risk {risk:.0f}$ bütçeyi aştı"


def test_nsp_position_value_respects_type_cap():
    """Pozisyon değeri tip tavanını (A = %25 = 2500$) aşmamalı — canlıda 141.830$'dı."""
    size = _authoritative_position_size(_req(position_size=2901))
    assert size * 48.89 <= 10000 * 0.25 + 50.0


# ── Meşru davranış korunuyor ─────────────────────────────────────────────

def test_smaller_client_size_is_respected():
    """Kullanıcı bilinçli olarak daha küçük pozisyon isteyebilir."""
    assert _authoritative_position_size(_req(position_size=10)) == 10


def test_missing_client_size_falls_back_to_server():
    """position_size gönderilmezse sunucu kendi hesabını kullanır."""
    assert _authoritative_position_size(_req(position_size=0)) == 37


def test_larger_portfolio_allows_larger_size():
    """Taban büyürse boyut da büyür — tavan portföye göreli, sabit değil."""
    small = _authoritative_position_size(_req(position_size=99999, portfolio_value=10_000))
    big = _authoritative_position_size(_req(position_size=99999, portfolio_value=100_000))
    assert big > small
    assert big * 48.89 <= 100_000 * 0.25 + 50.0


def test_type_b_gets_tighter_cap_than_type_a():
    """
    Tip tavanları farklı (B %20 < A/C %25) — tipe duyarlılık korunuyor.
    (Eski test Type S kullanıyordu; Type S 2026-08-04'te kaldırıldı — girdisi
    olan short-interest geçmişe dönük mevcut olmadığı için hiç doğrulanamamıştı.)
    """
    a = _authoritative_position_size(
        _req(position_size=99999, swing_type="A", entry_price=10.0, stop_loss=9.99))
    b = _authoritative_position_size(
        _req(position_size=99999, swing_type="B", entry_price=10.0, stop_loss=9.99))
    assert b < a


# ── Tutarsız girdi ───────────────────────────────────────────────────────

@pytest.mark.parametrize("entry,stop", [(10.0, 10.0), (10.0, 12.0)])
def test_stop_not_below_entry_is_rejected(entry, stop):
    """entry <= stop → risk tanımsız; 0 dönmeli (endpoint bunu invalid_risk'e çevirir)."""
    assert _authoritative_position_size(
        _req(entry_price=entry, stop_loss=stop, position_size=500)) == 0
