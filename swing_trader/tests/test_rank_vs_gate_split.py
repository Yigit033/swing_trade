"""
SIRALAMA ile BARAJ ayrımı (2026-08-05).

VCE işaretleri (+8 premium, +5 tight-coil) ölçülmüş gerçek bilgi taşıyor: aynı
sinyal sayısında bonuslu skorla sıralamak bonussuzdan daha iyi sonuç veriyor
(n=71'de EV +6.33% vs +5.77%). AMA eskiden tek bir `quality_score`'a eklendiği
için hem SIRALAMADA hem BARAJDA kullanılıyorlardı — ve baraj tarafında 34
sinyali Q80'in üstüne taşıyorlardı (o 34'ün EV'si +0.89%, taban +4.19%).

Para ölçümü (scripts/measure_threshold_money.py, slot-kısıtlı portföy, maliyet
dahil) ayrımın kârlı olduğunu gösterdi — canlıda tip tavanı %20-25 olduğu için
4-5 eşzamanlı pozisyon sığıyor:
    slot 3 : +53.7% → +104.1%   |  slot 4 : +47.5% → +76.0%
    slot 5 : +39.3% →  +57.5%   |  slot 8 : +39.2% → +50.6%

Bu test ayrımın geri birleşmesini engeller.
"""

import inspect
import re

from swing_trader.small_cap import engine as engine_mod

SRC = inspect.getsource(engine_mod)


def test_vce_marks_do_not_touch_quality_score():
    """
    `quality_score += 8` / `+= 5` geri gelmemeli. Gelirse baraj yine kayar ve
    zayıf sinyaller eşiği geçer (fırsat maliyeti kârı yer).
    """
    assert not re.search(r"quality_score\s*\+=", SRC), (
        "VCE işareti doğrudan quality_score'a ekleniyor — baraj kayar. "
        "İşaretler yalnız rank_score'a girmeli."
    )


def test_rank_score_is_quality_plus_marks():
    assert re.search(r"rank_score\s*=\s*quality_score\s*\+\s*_rank_bonus", SRC), (
        "rank_score = quality_score + işaretler tanımı bozulmuş"
    )


def test_gate_compares_raw_quality_score():
    """Rejim tabanı HAM skora uygulanmalı (rank_score'a DEĞİL)."""
    m = re.search(r"if\s+(\w+)\s*<\s*_type_min_q", SRC)
    assert m, "rejim tabanı karşılaştırması bulunamadı"
    assert m.group(1) == "quality_score", (
        f"baraj '{m.group(1)}' ile karşılaştırıyor — ham skor olmalı"
    )


def test_sorting_uses_rank_score():
    """Sıralama işaretleri KULLANMALI — bilgiyi çöpe atmıyoruz."""
    m = re.search(r"signals\.sort\(\s*key=lambda[^)]*?rank_score", SRC, re.S)
    assert m, "sıralama rank_score kullanmıyor — ölçülmüş sıralama avantajı kayıp"


def test_both_scores_reach_the_signal_dict():
    """
    İkisi de çıktıya girmeli: quality_score (baraj/gösterim) ve rank_score
    (sıralama). Biri düşerse ya UI yanlış sıralar ya ölçüm yanlış okur —
    2026-08-05'te vce bayraklarında tam bu olmuştu.
    """
    for key in ("'quality_score':", "'rank_score':", "'rank_bonus':"):
        assert key in SRC, f"{key} sinyal sözlüğünde yok"


def test_api_filters_on_quality_not_rank():
    """
    API katmanı da (effective_min_quality) ham skora bakmalı; aksi halde motor
    içi taban ile API tabanı farklı skorlara uygulanır ve eşik anlamı bölünür.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "api" / "routers" / "scanner.py").read_text(
        encoding="utf-8"
    )
    m = re.search(r"\[\s*s\s+for\s+s\s+in\s+signals\s+if\s+s\.get\(\s*['\"](\w+)['\"]", src)
    assert m, "API kalite filtresi bulunamadı"
    assert m.group(1) == "quality_score", (
        f"API '{m.group(1)}' ile filtreliyor — ham quality_score olmalı"
    )
