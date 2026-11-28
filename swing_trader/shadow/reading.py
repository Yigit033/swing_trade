"""Ek Y — GRW sinyallerine "okuma notu": şirketi okuyan bir dil modeli AL ya da ATLA der (yalnız kayda yazılır).

GRW'nin girişini, çıkışını, boyutunu DEĞİŞTİRMEZ. Ön kayıt: lab/PREREGISTRATION.md Ek Y. Talimat metni (``READ_PROMPT``)
sabittir; değişirse ``READ_PROMPT_SHA256`` değişir ve kayıtlar ayrı tutulur (yeni ek gerekir).

Geçmiş veriyle sınanmaz: model geçmişte ne olduğunu biliyor olabilir. Not, girişin açılışından önce yazılır.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

READ_PROMPT = """Sen deneyimli bir büyüme hissesi analistisin. Sana bir ABD hissesinin şirket kartı verilecek.
Bu hisse bugün teknik bir kırılım yaptı (fiyat 20 günün zirvesini yüksek hacimle aştı) ve son çeyrek satışı bir yıl
öncesine göre en az %25 arttı, net kâr/zarar da bir yıl öncesinden iyi.

Görevin: büyümenin gerçek ve sürdürülebilir olup olmadığını, şirketin kalitesini ve kırmızı bayrakları
değerlendirip bu hisseyi önümüzdeki haftalarda tutmak için ALMAYA DEĞER mi, karar vermek.

Mutlaka kontrol et:
- Satış tabanı anlamlı mı? Çok küçük satış üzerinden hesaplanan büyüme yüzdesi yanıltıcıdır.
- Büyüme tek seferlik mi (satın alma/birleşme, tek müşteri, tek seferlik lisans ya da hibe geliri)?
- Net zarar satışa göre büyük mü, şirket nakit yakıyor mu?
- Haberlerde hisse ihracı/seyreltme, dava, düzenleyici risk, muhasebe sorunu ya da satın alınma teklifi
  (fiyat teklife sabitlenir, yükseliş sınırlı) var mı?
- Sektörün ve ürünün arkasında gerçek bir talep var mı?

Yalnız verilen karta ve kartın tarihine kadarki genel bilgine dayan; kartın tarihinden sonrasını bildiğini varsayma,
bilmediğin şeyi uydurma. Kararsız kalırsan ve ciddi bir kırmızı bayrak yoksa AL de; ATLA yalnız somut bir neden
varsa.

Yanıtın YALNIZ şu JSON olsun, başka hiçbir metin yazma:
{"verdict": "AL" ya da "ATLA", "reasons": ["en çok 3 kısa Türkçe madde"], "red_flags": ["en çok 3 kısa Türkçe madde, yoksa boş liste"]}"""

READ_PROMPT_SHA256 = hashlib.sha256(READ_PROMPT.encode("utf-8")).hexdigest()
SUMMARY_CHARS = 1200
MAX_NEWS = 10


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _news_items(raw, now: datetime) -> List[Dict]:
    """yfinance haber listesi (eski ve yeni biçim) → [{title, publisher, published}] (yalnız ``now``dan önce)."""
    out = []
    for n in raw or []:
        c = n.get("content") if isinstance(n.get("content"), dict) else n
        title = c.get("title")
        pub = c.get("pubDate") or c.get("displayTime")
        ts = None
        if pub:
            try:
                ts = datetime.fromisoformat(str(pub).replace("Z", "+00:00"))
            except ValueError:
                ts = None
        elif c.get("providerPublishTime"):
            ts = datetime.fromtimestamp(int(c["providerPublishTime"]), tz=timezone.utc)
        if not title or ts is None or ts > now:
            continue
        prov = c.get("provider")
        publisher = prov.get("displayName") if isinstance(prov, dict) else (c.get("publisher") or "")
        out.append(dict(title=str(title)[:200], publisher=str(publisher or "")[:60], published=ts.date().isoformat()))
    out.sort(key=lambda x: x["published"], reverse=True)
    return out[:MAX_NEWS]


def build_card(ticker: str, asof: date, details: Dict, profile: Dict, now: datetime) -> Dict:
    """Ön kayıttaki sabit girdi listesi → modelin göreceği kart."""
    info = (profile or {}).get("info") or {}
    g = (details or {}).get("growth") or {}
    return dict(
        tarih=asof.isoformat(), hisse=ticker,
        sirket=info.get("longName") or info.get("shortName"), sektor=info.get("sector"), sanayi=info.get("industry"),
        is_tanimi=(info.get("longBusinessSummary") or "")[:SUMMARY_CHARS] or None,
        piyasa_degeri_usd=_num(info.get("marketCap")) or _num((details or {}).get("cap")),
        calisan=_num(info.get("fullTimeEmployees")),
        buyume=dict(ceyrek_sonu=g.get("quarter_end"), satis_usd=_num(g.get("rev")), satis_bir_yil_once_usd=_num(g.get("rev_prev")),
                    satis_buyumesi_yuzde=_num(g.get("sales_growth_pct")), net_kar_zarar_usd=_num(g.get("ni")),
                    net_kar_zarar_bir_yil_once_usd=_num(g.get("ni_prev"))),
        fiyat=dict(sinyal_kapanis=_num((details or {}).get("close")), kirilim_gunu_dibi=_num((details or {}).get("signal_low")),
                   getiri_12_ay_yuzde=round(100 * _num((details or {}).get("ret_12m")), 1)
                   if _num((details or {}).get("ret_12m")) is not None else None),
        haberler=_news_items((profile or {}).get("news"), now),
    )


def parse_verdict(text: Optional[str]) -> Optional[Dict]:
    """Model yanıtı → {verdict, reasons, red_flags}; geçersizse None."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    v = str(d.get("verdict", "")).strip().upper()
    if v not in ("AL", "ATLA"):
        return None
    clip = lambda xs: [str(x)[:200] for x in (xs if isinstance(xs, list) else [])][:3]
    return dict(verdict=v, reasons=clip(d.get("reasons")), red_flags=clip(d.get("red_flags")))


def read_signal(ticker: str, asof: date, details: Dict, now: datetime, profile: Dict, client=None) -> Dict:
    """Tek okuma. Dönüş: status OK (+ verdict, reasons, red_flags) ya da NO_LLM / PARSE_ERROR; her durumda
    provider, model, prompt_sha256 ve kartın özeti (card_sha256)."""
    if client is None:
        from ..genai.llm_client import LLMClient
        client = LLMClient()
    card = build_card(ticker, asof, details, profile, now)
    body = json.dumps(card, ensure_ascii=False, default=str)
    out = dict(provider=getattr(client, "provider", None), model=getattr(client, "model", None),
               prompt_sha256=READ_PROMPT_SHA256, card_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
               news=len(card["haberler"]), has_summary=bool(card["is_tanimi"]))
    if not getattr(client, "available", False):
        return dict(out, status="NO_LLM")
    text = client.complete("Şirket kartı:\n" + body, system_prompt=READ_PROMPT, max_tokens=600, temperature=0.0)
    v = parse_verdict(text)
    if v is None:
        return dict(out, status="PARSE_ERROR", raw=(text or "")[:300])
    return dict(out, status="OK", **v)


def reading_report(trades: List[Dict], min_closed: int = 40, min_group: int = 10) -> Dict:
    """Ek Y değerlendirmesi (yalnız zamanında okunmuş, geç kararsız, ısınmasız GRW kayıtları)."""
    rows = [t for t in trades if t.get("strategy") == "GRW" and t.get("status") in ("PENDING", "OPEN", "CLOSED")]
    rd = lambda t: ((t.get("details") or {}).get("reading") or {}) if isinstance(t.get("details"), dict) else {}
    ok = [t for t in rows if rd(t).get("status") == "OK" and not rd(t).get("late")
          and not t.get("late_decision") and not t.get("warmup")]
    groups = {}
    for v in ("AL", "ATLA"):
        g = [t for t in ok if rd(t).get("verdict") == v]
        closed = [float(t["net_pct"]) for t in g if t.get("status") == "CLOSED" and t.get("net_pct") is not None]
        groups[v] = dict(n=len(g), closed=len(closed), mean_net=round(sum(closed) / len(closed), 3) if closed else None)
    n_closed = groups["AL"]["closed"] + groups["ATLA"]["closed"]
    ready = n_closed >= min_closed and min(groups["AL"]["closed"], groups["ATLA"]["closed"]) >= min_group
    status, y1, y2 = "TOPLANIYOR", None, None
    if ready:
        y1 = groups["AL"]["mean_net"] > groups["ATLA"]["mean_net"]
        y2 = groups["ATLA"]["mean_net"] < 0
        status = "GEÇTİ" if (y1 and y2) else "KALDI"
    elif n_closed >= 60 and groups["ATLA"]["closed"] < min_group:
        status = "KALDI"                                   # okuma ayırt etmiyor
    return dict(groups=groups, read=len([t for t in rows if rd(t).get("status") == "OK"]),
                late=len([t for t in rows if rd(t).get("status") == "OK" and rd(t).get("late")]),
                missing=len([t for t in rows if rd(t).get("status") != "OK"]), closed=n_closed,
                need_closed=min_closed, need_group=min_group, Y1=y1, Y2=y2, status=status,
                prompt_sha256=READ_PROMPT_SHA256, deadline="2028-06-30")
