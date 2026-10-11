"""Descriptive gap attribution; forward returns are NOT trade expectancy."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from ..master import write_json, digest
from .features import VCE_PREDICATES, RVOL_PREDICATES


def events(df, mask):
    chosen=[]
    for _, group in df.loc[mask].sort_values(['ticker','session_index']).groupby('ticker',sort=True):
        last=-1000
        for i, row in group.iterrows():
            if row.session_index > last+30:
                chosen.append(i)
                last=row.session_index
    return df.loc[chosen]


def cluster_ci(df, value, cluster='ticker', iterations=1000):
    g=df.dropna(subset=[value]).groupby(cluster)[value].agg(['sum','count'])
    if len(g)<2:
        return None
    rng=np.random.default_rng(7301)
    sums, counts=g['sum'].to_numpy(),g['count'].to_numpy()
    estimates=[]
    for _ in range(iterations):
        idx=rng.integers(0,len(g),len(g))
        estimates.append(sums[idx].sum()/counts[idx].sum())
    return [float(x) for x in np.percentile(estimates,[2.5,97.5])]


def difference_ci(a,b,value='net20_50',cluster='ticker'):
    aa=a.dropna(subset=[value]).groupby(cluster)[value].agg(['sum','count'])
    bb=b.dropna(subset=[value]).groupby(cluster)[value].agg(['sum','count'])
    ix=aa.index.union(bb.index)
    if len(aa)<2 or len(bb)<2:
        return None
    aa,bb=aa.reindex(ix,fill_value=0).to_numpy(),bb.reindex(ix,fill_value=0).to_numpy()
    rng=np.random.default_rng(7301)
    results=[]
    for _ in range(1000):
        draw=rng.integers(0,len(ix),len(ix))
        av,bv=aa[draw].sum(axis=0),bb[draw].sum(axis=0)
        if av[1] and bv[1]:
            results.append(av[0]/av[1]-bv[0]/bv[1])
    return [float(x) for x in np.percentile(results,[2.5,97.5])] if results else None


def distributions(series):
    numeric=pd.to_numeric(series,errors='coerce')
    x=numeric.loc[np.isfinite(numeric)]
    return None if x.empty else {'n':len(x),'nonfinite_or_missing':len(series)-len(x),
        'mean':float(x.mean()),'median':float(x.median()),
        'p10':float(x.quantile(.1)),'p90':float(x.quantile(.9)),'min':float(x.min()),'max':float(x.max())}


def stats(raw):
    ev=events(raw,pd.Series(True,index=raw.index)) if len(raw) else raw
    result={'signal_count':len(raw),'independent_events':len(ev),'tickers':int(ev.ticker.nunique()),
            'eligibility':{'VERIFIED_ELIGIBLE':0,'UNKNOWN_HISTORICAL_CAP_FLOAT_EARNINGS':len(ev)},
            'horizons':{},'liquidity_musd':distributions(ev.dv/1e6), 'atr_pct':distributions(ev.atr_pct),
            'gap_abs_gt5_count':int((ev.entry_gap_pct.abs()>5).sum()),
            'zero_volume_history_caution':int((ev.zero_volume52>0).sum()),
            'regimes':{}, 'temporal_block_inference':'INSUFFICIENT_NONOVERLAPPING_30_SESSION_BLOCKS_IN_30_SESSION_EVALUATION'}
    for n in (5,10,20,30):
        valid=ev.dropna(subset=[f'r{n}'])
        result['horizons'][str(n)]={'n':len(valid),'missing_or_unreliable_outcomes':len(ev)-len(valid),
            'gross':distributions(valid[f'r{n}']),
            'positive_rate_pct':float((valid[f'r{n}']>0).mean()*100) if len(valid) else None,
            'mfe':distributions(valid[f'mfe{n}']),'mae':distributions(valid[f'mae{n}']),
            'net':{str(bps):distributions(valid[f'net{n}_{bps}']) for bps in (20,50,100)},
            'close_ge_pct':{str(t):int((valid[f'r{n}']>=t).sum()) for t in (10,20,30)},
            'mfe_ge_pct':{str(t):int((valid[f'mfe{n}']>=t).sum()) for t in (10,20,30)}}
    result['ticker_ci_net20_50']=cluster_ci(ev,'net20_50')
    result['date_ci_sensitivity_net20_50']=cluster_ci(ev,'net20_50','date')
    result['net20_exclude_best3_events']=distributions(ev.sort_values('net20_50',ascending=False).iloc[3:].net20_50)
    top=ev.groupby('ticker').net20_50.mean().nlargest(3).index
    result['net20_exclude_best3_tickers']=distributions(ev.loc[~ev.ticker.isin(top),'net20_50'])
    for regime,group in ev.groupby('spy_context_not_full_regime'):
        result['regimes'][regime]={'events':len(group),'net20_50':distributions(group.net20_50),
                                  'context_only_not_full_market_regime':True}
    return result


def summarize(output):
    if not (output/'complete.json').exists():
        raise ValueError('INCOMPLETE_PANEL')
    frames, details=[],{}
    parity=0
    for path in sorted((output/'panels').glob('*.csv.gz')):
        det=json.loads((output/'details'/(path.name.split('.')[0]+'.json')).read_text())
        if digest(path.read_bytes())!=det['panel_sha256']:
            raise ValueError('PANEL_CHECKSUM_MISMATCH')
        frames.append(pd.read_csv(path))
        parity+=det['parity_checks']
        for row in det['details']:
            details[(row['ticker'],row['date'])]=row
    df=pd.concat(frames,ignore_index=True)
    flags=['history','base','data_missing','price_pass','liquidity_pass','champion','vce','rvol','prebreak','relaunch']+VCE_PREDICATES+RVOL_PREDICATES
    flags += ['drop_'+p for p in VCE_PREDICATES+RVOL_PREDICATES]
    for col in flags:
        df[col]=df[col].fillna(False).astype(bool)
    dates=sorted(df.date.unique())
    df['session_index']=df.date.map({d:i for i,d in enumerate(dates)})
    trigger_sessions={tk:set(g.loc[g.champion,'session_index']) for tk,g in df.groupby('ticker')}
    report={'cohort':'HISTORICAL_RESEARCH_SURVIVORSHIP_BIASED_NOT_INDEPENDENT_FORWARD',
        'date_range':[dates[0],dates[-1]],'sessions':len(dates),'ticker_dates':len(df),
        'direct_original_predicate_parity_checks':parity,'parity_mismatches':0,
        'daily_coverage':df.groupby('date')[['history','base','data_missing']].sum().astype(int).to_dict('index'),
        'phases':{},'attribution':{},'predicate_ablation':{},'opportunity_thresholds':{},
        'detail_rows':len(details),'full_trade_eligibility_verified':0}
    for phase in ('RESEARCH','PURGE','EVALUATION'):
        subset=df.loc[df.phase==phase]
        base=subset.loc[subset.base]
        methods={'BASELINE':base, 'CHAMPION':base.loc[base.champion],
                 'PREBREAK':base.loc[base.prebreak], 'RELAUNCH':base.loc[base.relaunch],
                 'PREBREAK_ADDED':base.loc[base.prebreak & ~base.champion],
                 'RELAUNCH_ADDED':base.loc[base.relaunch & ~base.champion]}
        record={'dates':[subset.date.min(),subset.date.max()],'base_ticker_dates':len(base),
                'methods':{k:stats(v) for k,v in methods.items()}}
        champion_ev=events(methods['CHAMPION'],pd.Series(True,index=methods['CHAMPION'].index))
        for name in ('PREBREAK','RELAUNCH','PREBREAK_ADDED','RELAUNCH_ADDED'):
            raw=methods[name]
            ev=events(raw,pd.Series(True,index=raw.index))
            record['methods'][name]['new_tickers_vs_champion_in_phase']=len(set(raw.ticker)-set(methods['CHAMPION'].ticker))
            record['methods'][name]['difference_ci_vs_champion_ticker']=difference_ci(ev,champion_ev)
            record['methods'][name]['difference_ci_vs_champion_date']=difference_ci(ev,champion_ev,cluster='date')
        report['phases'][phase]=record
    # Descriptive labels are explicitly allowed to see outcomes; selectors never do.
    for n in (5,10,20,30):
        for t in (10,20,30):
            strong=events(df,df.base & (df[f'r{n}']>=t))
            report['opportunity_thresholds'][f'close{n}_ge{t}']={'events':len(strong),
                'trigger_at_event_start':int(strong.champion.sum()),
                'trigger_within_cluster_after_start':sum(any(r.session_index<=i<=r.session_index+30 for i in trigger_sessions[r.ticker]) for r in strong.itertuples()) if n==20 and t==10 else None}
            mfe=events(df,df.base & (df[f'mfe{n}']>=t))
            report['opportunity_thresholds'][f'mfe{n}_ge{t}']={'events':len(mfe),'trigger_at_event_start':int(mfe.champion.sum())}
    strong=events(df,df.base & (df.r20>=10))
    cats=Counter()
    gate_fails=Counter()
    groups={}
    for row in strong.itertuples():
        det=details.get((row.ticker,row.date))
        if not det:
            raise ValueError('MISSING_OPPORTUNITY_DETAIL')
        failures=[k for k in ('swing','rsi','weinstein','quality') if det['stages'][k]['status']=='FAIL']
        gate_fails.update(failures)
        category='3_TRIGGER_DOWNSTREAM_FAIL' if row.champion and failures else '5_UNKNOWN_FUNDAMENTALS'
        cats[category]+=1
        subgroup=('TRIGGER' if row.champion else 'NO_TRIGGER')+('_KNOWN_DOWNSTREAM_FAIL' if failures else '_KNOWN_DOWNSTREAM_CLEAR')
        groups.setdefault(subgroup,[]).append(row.Index)
    report['attribution']={'primary_opportunities':len(strong),'mutually_exclusive':dict(cats),
        'verified_category1_all_gates_pass':0,'verified_category4_no_trigger_other_gates_pass':0,
        'unknown_not_zero_true_eligible':'HISTORICAL_FUNDAMENTALS_MISSING',
        'known_downstream_failures_nonexclusive':dict(gate_fails),
        'orthogonal_groups':{k:len(v) for k,v in groups.items()},
        'outside_price_liquidity_or_history_strong_events':len(events(df,~df.base & (df.r20>=10))),
        'feature_distributions':{name:{c:distributions(frame[c]) for c in ['squeeze','rvol50','change5','change20','distance20high_pct','range_ratio','accumulation_ratio','dry_volume','atr_pct','dv']}
            for name,frame in [('strong',strong),('all_base_primary',events(df,df.base)),('negative_primary',events(df,df.base & (df.r20<=0)))]}}
    for pred in VCE_PREDICATES+RVOL_PREDICATES:
        missing=strong.loc[~strong[pred]]
        report['predicate_ablation'][pred]={'failed_primary_opportunities':len(missing),
            'failed_opportunity_net20_50':distributions(missing.net20_50),
            'added_all_outcomes_by_phase':{phase:stats(df.loc[(df.phase==phase)&df['drop_'+pred]]) for phase in ('RESEARCH','EVALUATION')}}
    # Strong/negative cases are descriptive examples, never training samples selected posthoc.
    examples=strong.sort_values('r20',ascending=False).head(20)
    report['examples']=[{'ticker':r.ticker,'date':r.date,'r20':r.r20,'mfe20':r.mfe20,'mae20':r.mae20,
        'champion':bool(r.champion),'details':details[(r.ticker,r.date)]} for r in examples.itertuples()]
    write_json(output/'summary_final.json',report)
    print(json.dumps({'summary':str(output/'summary_final.json'),'ticker_dates':len(df),'parity_checks':parity,
                      'primary_opportunities':len(strong),'attribution':dict(cats)},indent=2))
