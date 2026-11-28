"""Ek Y — GRW okuma notu (AL / ATLA): yalnız kayda yazılır, GRW kararlarını değiştirmez."""
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from swing_trader.shadow import reading, runner
from swing_trader.shadow.ledger import ShadowLedger
from swing_trader.tests.test_shadow_grw import ASOF, OK, _hit, _run
from swing_trader.tests.test_shadow_runner import FakeSrc

NY = ZoneInfo("America/New_York")
NOW = datetime(2025, 6, 16, 11, 0, tzinfo=timezone.utc)                    # 07:00 ET


class FakeLLM:
    provider, model, available = "fake", "fake-1", True

    def __init__(self, text):
        self.text, self.calls = text, []

    def complete(self, prompt, system_prompt="", max_tokens=0, temperature=0.5):
        self.calls.append(dict(prompt=prompt, system=system_prompt, temperature=temperature))
        return self.text


PROFILE = dict(info=dict(longName="Abc Corp", sector="Technology", industry="Software",
                         longBusinessSummary="x" * 5000, marketCap=5e9, fullTimeEmployees=900),
               news=[dict(content=dict(title="Abc wins contract", pubDate="2025-06-12T14:00:00Z",
                                       provider=dict(displayName="Reuters"))),
                     dict(content=dict(title="from the future", pubDate="2025-06-20T14:00:00Z")),
                     dict(title="old format", publisher="AP", providerPublishTime=1749600000)])
DETAILS = dict(close=22.0, signal_low=21.8, ret_12m=0.5, cap=5e9,
               growth=dict(OK, rev=160e3, rev_prev=23e3, ni=-4.8e6, ni_prev=-5.5e6))


def test_card_uses_only_the_preregistered_inputs_and_drops_future_news():
    card = reading.build_card("ABC", ASOF, DETAILS, PROFILE, NOW)
    assert len(card["is_tanimi"]) == reading.SUMMARY_CHARS and card["sirket"] == "Abc Corp"
    assert card["buyume"]["satis_usd"] == 160e3 and card["buyume"]["net_kar_zarar_usd"] == -4.8e6
    assert card["fiyat"] == dict(sinyal_kapanis=22.0, kirilim_gunu_dibi=21.8, getiri_12_ay_yuzde=50.0)
    assert [n["title"] for n in card["haberler"]] == ["Abc wins contract", "old format"]     # gelecek haber yok


@pytest.mark.parametrize("text,want", [
    ('{"verdict": "ATLA", "reasons": ["satış yok denecek kadar az"], "red_flags": ["a", "b", "c", "d"]}', "ATLA"),
    ('Tamam.\n```json\n{"verdict": "al", "reasons": [], "red_flags": []}\n```', "AL"),
    ('{"verdict": "BELKİ"}', None), ("cevap yok", None), (None, None)])
def test_parse_verdict(text, want):
    v = reading.parse_verdict(text)
    assert (v or {}).get("verdict") == want
    if v:
        assert len(v["red_flags"]) <= 3


def test_read_signal_records_model_prompt_hash_and_temperature_zero():
    llm = FakeLLM('{"verdict": "ATLA", "reasons": ["satış tabanı çok küçük"], "red_flags": ["nakit yakıyor"]}')
    r = reading.read_signal("ABC", ASOF, DETAILS, NOW, profile=PROFILE, client=llm)
    assert (r["status"], r["verdict"], r["provider"], r["model"]) == ("OK", "ATLA", "fake", "fake-1")
    assert r["prompt_sha256"] == reading.READ_PROMPT_SHA256 and llm.calls[0]["temperature"] == 0.0
    assert llm.calls[0]["system"] == reading.READ_PROMPT and '"satis_usd": 160000.0' in llm.calls[0]["prompt"]
    llm.available = False
    assert reading.read_signal("ABC", ASOF, DETAILS, NOW, profile=PROFILE, client=llm)["status"] == "NO_LLM"
    llm.available, llm.text = True, "anlamsız"
    assert reading.read_signal("ABC", ASOF, DETAILS, NOW, profile=PROFILE, client=llm)["status"] == "PARSE_ERROR"


class ReadingSrc(FakeSrc):
    def __init__(self, verdicts):
        super().__init__()
        self.verdicts, self.read_calls = verdicts, []

    def read_signal(self, ticker, asof, details, now):
        self.read_calls.append(ticker)
        v = self.verdicts.get(ticker)
        if isinstance(v, Exception):
            raise v
        return dict(status="OK", verdict=v, reasons=["r"], red_flags=[]) if v else dict(status="NO_LLM")


def test_runner_reads_pending_grw_signals_without_changing_the_rule(tmp_path):
    L = ShadowLedger(str(tmp_path / "s.db"))
    src = ReadingSrc({"ABC": "ATLA", "XYZ": "AL"})
    src.grw = {ASOF: [_hit("ABC", 0.9), _hit("XYZ", 0.4), _hit("NOD", 0.2)]}
    src.growth = {"ABC": OK, "XYZ": OK}                                        # NOD: veri yok → SKIPPED, okunmaz
    out = _run(L, src, 2025, 6, 16)
    assert out["GRW"]["PENDING"] == 2 and out["GRW_READ"] == dict(todo=2, ok=2)
    rows = {t["ticker"]: t for t in L.trades("GRW")}
    assert rows["ABC"]["status"] == "PENDING"                                  # ATLA olsa da GRW yine alır
    assert rows["ABC"]["details"]["reading"]["verdict"] == "ATLA" and rows["ABC"]["details"]["reading"]["late"] == 0
    assert rows["ABC"]["details"]["growth"] == OK                              # mevcut ayrıntılar korunur
    assert "reading" not in (rows["NOD"]["details"] or {}) and sorted(src.read_calls) == ["ABC", "XYZ"]
    _run(L, src, 2025, 6, 16, hour=8)                                          # yeniden koşu: tekrar okumaz
    assert sorted(src.read_calls) == ["ABC", "XYZ"]
    rep = L.get_meta("report")["reading"]
    assert rep["groups"]["ATLA"]["n"] == 1 and rep["groups"]["AL"]["n"] == 1 and rep["status"] == "TOPLANIYOR"
    assert not [i for i in runner.run_health(L, now=datetime(2025, 6, 16, 8, 5, tzinfo=NY))["issues"] if "READ" in i]


def test_failed_reading_is_retried_and_late_reading_is_flagged(tmp_path):
    L = ShadowLedger(str(tmp_path / "s.db"))
    src = ReadingSrc({"ABC": RuntimeError("llm down")})
    src.grw, src.growth = {ASOF: [_hit("ABC")]}, {"ABC": OK}
    out = _run(L, src, 2025, 6, 16)
    assert out["GRW_READ"] == dict(todo=1, ok=0, ERROR=1)
    assert L.trades("GRW")[0]["details"]["reading"]["status"] == "ERROR"
    # okuma eksikliği sağlığı bozmaz → yedek tetikleyici koşmaz; elle koşu açılıştan sonra okursa "geç"
    assert runner.run_health(L, now=datetime(2025, 6, 16, 7, 5, tzinfo=NY))["complete"] is True
    src.verdicts["ABC"] = "AL"
    _run(L, src, 2025, 6, 16, hour=11)
    r = L.trades("GRW")[0]["details"]["reading"]
    assert (r["status"], r["verdict"], r["late"]) == ("OK", "AL", 1)
    assert L.get_meta("report")["reading"]["late"] == 1 and L.get_meta("report")["reading"]["groups"]["AL"]["n"] == 0


def _closed(i, verdict, net, late=0):
    return dict(strategy="GRW", status="CLOSED", net_pct=net, late_decision=0, warmup=0,
                details=dict(reading=dict(status="OK", verdict=verdict, late=late)))


def test_reading_report_gate():
    al = [_closed(i, "AL", 2.0) for i in range(30)]
    atla = [_closed(i, "ATLA", -1.0) for i in range(10)]
    rep = reading.reading_report(al + atla)
    assert (rep["status"], rep["Y1"], rep["Y2"], rep["closed"]) == ("GEÇTİ", True, True, 40)
    rep = reading.reading_report(al + [_closed(i, "ATLA", 0.5) for i in range(10)])   # elenenler kazandırdı
    assert (rep["status"], rep["Y2"]) == ("KALDI", False)
    assert reading.reading_report(al + atla[:9])["status"] == "TOPLANIYOR"              # ATLA grubu < 10
    assert reading.reading_report([_closed(i, "AL", 1.0) for i in range(60)])["status"] == "KALDI"
    late = reading.reading_report(al + [_closed(i, "ATLA", -1.0, late=1) for i in range(10)])
    assert late["groups"]["ATLA"]["n"] == 0 and late["late"] == 10


def test_prompt_hash_is_pinned():
    """Talimat değişirse bu test düşer: değişiklik yeni ön kayıt eki ister (Ek Y)."""
    assert reading.READ_PROMPT_SHA256 == PINNED


PINNED = "4c13db191222c3c825bfaecafc9248e21c622529c6f7bf559fd0d63e997b5983"


def test_api_exposes_only_the_reading_summary():
    from api.routers.shadow import _reading
    t = dict(details=dict(growth=OK, reading=dict(status="OK", verdict="AL", reasons=["a"], red_flags=[], late=0,
                                                  card_sha256="x", prompt_sha256="y", raw="z")))
    assert _reading(t) == dict(status="OK", verdict="AL", reasons=["a"], red_flags=[], late=0)
    assert _reading(dict(details=None)) is None and _reading(dict(details=dict(growth=OK))) is None
