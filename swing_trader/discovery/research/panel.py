"""Checkpointed offline cross-sectional EOD panels. No production writes."""
import json
import time
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
from dateutil.relativedelta import relativedelta
from ..bars import read_object, object_id
from ..master import write_json, digest, utc_now
from ..calendar import sessions
from ..shadow import evaluate, code_manifest
from ..bars import MODE
from swing_trader.small_cap.settings_config import SmallCapSettings
from swing_trader.small_cap.signals import SmallCapSignals
from swing_trader.small_cap.regime_logic import regime_unknown, regime_from_spy_close
from .features import features, outcomes, decision_frame, VCE_PREDICATES, RVOL_PREDICATES
from .protocol import PROTOCOL


def independent_indices(frame, mask):
    chosen, last = [], -10**9
    for i in range(len(frame)):
        if bool(mask.iloc[i]) and i > last+30:
            chosen.append(i)
            last = i
    return chosen


def selected_details(record, obj, day, settings, regime):
    df = decision_frame(obj, day)
    rows = [{"Date": str(t.date()), "Open": float(r.Open), "High": float(r.High),
             "Low": float(r.Low), "Close": float(r.Close), "Adj Close": float(r.Close),
             "Volume": float(r.Volume), "Dividends": 0., "Stock Splits": 0.} for t,r in df.iterrows()]
    reconstructed = {**obj, "rows": rows, "mode": MODE}
    row = evaluate(record, reconstructed, day, settings, regime, {}, "2026-10-11T00:00:00+00:00")
    return {"ticker": row["ticker"], "date": str(day), "stages": row["stages"],
            "quality_score": row["quality_score"], "rank_score": row["rank_score"],
            "quality_range": row.get("quality_range_given_unknown_float"),
            "rejections": row.get("rejection_reasons", []), "final_status": row["final_status"],
            "full_trade_eligibility": "UNKNOWN_HISTORICAL_FUNDAMENTALS",
            "note": "CURRENT_SECURITY_IDENTITY_NOT_HISTORICAL;NO_CURRENT_FACTS_BACKFILLED"}


def run(inputs, prep, cache, inventory, output):
    if inventory["protocol"] != PROTOCOL or inventory["preparation_id"] != object_id(prep):
        raise ValueError("PREREGISTRATION_INPUT_MISMATCH")
    if len(inventory["chosen_sessions"]) != 90:
        raise ValueError("INSUFFICIENT_SESSIONS_NO_CHALLENGER_RUN")
    output.mkdir(parents=True, exist_ok=True)
    source = code_manifest()
    root = Path(__file__).resolve().parents[3]
    source.update({str(p.relative_to(root)).replace('\\','/'):digest(p.read_bytes()) for p in Path(__file__).parent.glob('*.py')})
    spec = {"protocol": PROTOCOL, "inventory_id": object_id(inventory), "inputs_id": object_id(inputs),
            "preparation_id": object_id(prep), "sources": source}
    spec_path = output/'spec.json'
    if spec_path.exists():
        if json.loads(spec_path.read_text()) != spec:
            raise ValueError("RESUME_CODE_OR_PLAN_MISMATCH")
    else:
        write_json(spec_path, spec)
    (output/'panels').mkdir(exist_ok=True)
    (output/'details').mkdir(exist_ok=True)
    records = sorted((r for r in inputs['snapshot']['records'] if r['discovery_status']=='CANDIDATE'), key=lambda r:r['symbol'])
    refs = {o['symbol']:o['object_id'] for o in prep['outcomes'] if o.get('object_id')}
    start = min(inventory['first_dates'])
    calendar = pd.DatetimeIndex(sessions(date.fromisoformat(start), date.fromisoformat(prep['session'])))
    days = pd.DatetimeIndex(inventory['chosen_sessions'])
    settings = SmallCapSettings.model_validate(inputs['settings'])
    sig = SmallCapSignals(settings=settings)
    spy = read_object(cache, refs['SPY'])
    from ..bars import rows_frame
    spy_df, vix_df = rows_frame(spy['rows']), rows_frame(read_object(cache,refs['^VIX'])['rows'])
    regimes, contexts = {}, {}
    for day in days:
        sp = spy_df.loc[:day]
        contexts[day] = regime_from_spy_close(sp.Close)['regime'] if len(sp)>=204 else 'UNKNOWN_SHORT_SPY'
        regimes[day] = (regime_from_spy_close(sp.Close, float(vix_df.loc[day].Close))
                        if len(sp)>=204 and day in vix_df.index else regime_unknown('HISTORICAL_SPY_OR_VIX_MISSING'))
    begun = time.monotonic()
    for j, rec in enumerate(records):
        path, detpath = output/'panels'/f'{j:05}.csv.gz', output/'details'/f'{j:05}.json'
        if path.exists() and detpath.exists():
            continue
        if path.exists() or detpath.exists():
            raise ValueError('PARTIAL_CHECKPOINT_REQUIRES_NEW_RUN')
        symbol = rec['symbol']
        detail, parity = [], 0
        if symbol not in refs:
            panel = pd.DataFrame(index=days, data={'history':False, 'base':False, 'data_missing':True})
        else:
            obj = read_object(cache, refs[symbol])
            x, d = features(obj, calendar, settings)
            panel = x.join(outcomes(d)).reindex(days).copy()
            panel['data_missing'] = panel.price.isna()
            # Explicit diagnostic ablations: only added candidates, all losers included.
            for family, predicates in [('vce',VCE_PREDICATES),('rvol',RVOL_PREDICATES)]:
                for pred in predicates:
                    panel['drop_'+pred] = panel[[k for k in predicates if k!=pred]].all(axis=1) & ~panel.champion & panel.base
            # Direct originals on every positive and deterministic 1/90 negative per ticker.
            checks = panel.history & (panel.champion | (pd.Series(range(90),index=days)==j%90))
            for day in panel.index[checks]:
                frame = decision_frame(obj, day.date())
                frame = frame.loc[pd.Timestamp(day.date()+timedelta(days=1)-relativedelta(months=6)):]
                vce = sig.check_vce_breakout(frame)[0]
                rvol = sig.check_rvol_thrust(frame)[0]
                if (bool(vce),bool(rvol)) != (bool(panel.loc[day,'vce']),bool(panel.loc[day,'rvol'])):
                    raise ValueError(f'CHAMPION_PARITY_MISMATCH:{symbol}:{day.date()}')
                parity += 1
            # All signals plus outcome-defined independent opportunity starts, not all winners only.
            strong = panel.base & (panel.r20 >= 10)
            primary_opportunities = independent_indices(panel, strong)
            panel['opportunity_primary'] = False
            panel.loc[days[primary_opportunities], 'opportunity_primary'] = True
            selected = panel.history & (panel.champion | (panel.base & (panel.prebreak | panel.relaunch)) | panel.opportunity_primary)
            for day in panel.index[selected]:
                detail.append(selected_details(rec, obj, day.date(), settings, regimes[day]))
        panel['ticker'], panel['exchange'] = symbol, rec['exchange']
        panel['phase'] = ['RESEARCH']*30 + ['PURGE']*30 + ['EVALUATION']*30
        panel['regime'] = [regimes[t]['regime'] for t in days]
        panel['spy_context_not_full_regime'] = [contexts[t] for t in days]
        panel.index.name = 'date'
        panel.to_csv(path, compression='gzip', float_format='%.12g')
        write_json(detpath, {'ticker':symbol, 'parity_checks':parity, 'details':detail, 'panel_sha256':digest(path.read_bytes())})
        if (j+1)%100==0:
            print(json.dumps({'completed':j+1,'total':len(records),'elapsed_seconds':round(time.monotonic()-begun,1)}),flush=True)
    write_json(output/'complete.json',{'completed_at':utc_now(),'elapsed_seconds':time.monotonic()-begun,
        'spec_id':object_id(spec),'symbols':len(records),'decision_dates':[str(d.date()) for d in days]})
    print(json.dumps({'complete':True,'symbols':len(records),'elapsed_seconds':time.monotonic()-begun}))
