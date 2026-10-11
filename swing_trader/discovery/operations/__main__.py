"""Record frozen challenger hypotheses BEFORE reference entry; no network calls.

python -m swing_trader.discovery.operations --scan PHASE2_SCAN_DIR --cache CACHE --output NEW_BOOK.json
"""
import argparse
import json
from datetime import datetime, date, time, timezone
from pathlib import Path
import pandas as pd
from ..bars import object_id, read_object
from ..calendar import sessions, last_completed, next_session, NY
from ..master import write_json, digest
from ..opportunity import horizon_sessions
from ..research.features import features
from ..research.protocol import PROTOCOL
from swing_trader.small_cap.settings_config import SmallCapSettings


def observe(scan_dir, cache, previous=None, now=None):
    now=now or datetime.now(timezone.utc)
    scan=json.loads((scan_dir/'scan.json').read_text(encoding='utf-8'))
    inputs=json.loads((scan_dir/'inputs.json').read_text(encoding='utf-8'))
    manifest=json.loads((scan_dir/'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('complete'):
        raise ValueError('INCOMPLETE_SCAN')
    for name, expected in manifest['sha256'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or digest((scan_dir/name).read_bytes())!=expected:
            raise ValueError('INVALID_SCAN_MANIFEST')
    signal=date.fromisoformat(scan['inputs']['session'])
    entry=next_session(signal)
    if signal!=last_completed(now) or now>=datetime.combine(entry,time(9,30),NY):
        raise ValueError('NOT_CURRENT_PRE_ENTRY_PROSPECTIVE_OBSERVATION')
    if datetime.fromisoformat(scan['inputs']['decision_at'])>now:
        raise ValueError('SCAN_OBSERVED_AFTER_REGISTRATION')
    if previous and previous.get('book_kind')!='CHALLENGER_PROSPECTIVE_SHADOW':
        raise ValueError('DO_NOT_APPEND_TO_CHAMPION_BOOK')
    prior=list((previous or {}).get('observations',[]))
    if prior and max(r['signal_date'] for r in prior)>str(signal):
        raise ValueError('OUT_OF_ORDER_OBSERVATIONS')
    existing={r['observation_id'] for r in prior}
    config=SmallCapSettings.model_validate(inputs['settings'])
    objects={o['symbol']:o['object_id'] for o in inputs['preparation']['outcomes'] if o.get('object_id')}
    for row in scan['rows']:
        if row['ticker'] not in objects:
            continue
        obj=read_object(cache,objects[row['ticker']])
        if datetime.fromisoformat(obj['fetched_at'])>now:
            raise ValueError('CACHE_FROM_FUTURE')
        # Hard cutoff before computing any challenger decision features.
        obj={**obj,'rows':[r for r in obj['rows'] if r['Date']<=str(signal)]}
        if not obj['rows']:
            continue
        calendar=pd.DatetimeIndex(sessions(date.fromisoformat(obj['rows'][0]['Date']),signal))
        x,_=features(obj,calendar,config)
        value=x.iloc[-1]
        if not bool(value['base']):
            continue
        for name,column in [('PREBREAK_ACCUMULATION','prebreak'),('IMPULSE_BASE_RELAUNCH','relaunch')]:
            if not bool(value[column]):
                continue
            identity=object_id({'ticker':row['ticker'],'session':str(signal),'hypothesis':name,'protocol':PROTOCOL})
            if identity in existing:
                continue
            recent=[r for r in prior if r['ticker']==row['ticker'] and r['setup']==name and r['independent_until']>=str(signal)]
            prior.append({'observation_id':identity,'cohort':'PROSPECTIVE_SHADOW','setup':name,
                'ticker':row['ticker'],'signal_date':str(signal),'registered_at':now.isoformat(),
                'scan_id':scan['scan_id'],'universe_id':scan['inputs']['universe_id'],
                'source_bar_object_id':objects[row['ticker']], 'protocol_id':object_id(PROTOCOL),
                'entry_session':str(entry),'entry_price':None,'entry_method':'NEXT_SCHEDULED_SESSION_OPEN',
                'reference_close':float(value.price),'atr_percent':float(value.atr_pct),
                'avg_dollar_volume_20':float(value.dv),'legacy_membership':row['legacy_membership'],
                'broad_only_confirmed':row['legacy_membership']=='NO',
                'champion_trigger_same_day':row['trigger_positive'],
                'champion_context_stages_not_challenger_calibrated':row['stages'],
                'full_trade_eligibility':'NOT_ASSERTED_CHALLENGER_RESEARCH_ONLY',
                'independence_cluster':recent[0]['independence_cluster'] if recent else identity,
                'independent_observation':not recent,
                'independent_until':recent[0]['independent_until'] if recent else str(horizon_sessions(signal)[-1]),
                'future_status':'PENDING','not_trade_advice':True})
            existing.add(identity)
    files=[Path(__file__),Path(__file__).parents[1]/'research'/'features.py',Path(__file__).parents[1]/'research'/'protocol.py']
    return {'book_kind':'CHALLENGER_PROSPECTIVE_SHADOW','protocol':PROTOCOL,'observations':prior,
            'sources':{str(p.name):digest(p.read_bytes()) for p in files}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--previous',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    previous=json.loads(args.previous.read_text(encoding='utf-8')) if args.previous else None
    book=observe(args.scan,args.cache,previous)
    write_json(args.output,book)
    from collections import Counter
    print(json.dumps({'observations':len(book['observations']),'hypotheses':dict(Counter(r['setup'] for r in book['observations']))}))


if __name__=='__main__':
    main()
