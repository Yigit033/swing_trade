"""Offline supplementary diagnostics and immutable source/artifact sealing."""
import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
from ..master import digest, write_json
from ..research.analysis import events, stats, distributions, cluster_ci
from ..research.features import VCE_PREDICATES, RVOL_PREDICATES


def supplement(run):
    frames=[]
    detail={}
    for path in sorted((run/'panels').glob('*.csv.gz')):
        doc=json.loads((run/'details'/(path.name.split('.')[0]+'.json')).read_text())
        if digest(path.read_bytes())!=doc['panel_sha256']:
            raise ValueError('PANEL_INTEGRITY_ERROR')
        frames.append(pd.read_csv(path))
        detail.update({(r['ticker'],r['date']):r for r in doc['details']})
    df=pd.concat(frames,ignore_index=True)
    for c in ['base','history','champion','prebreak','relaunch','price_pass','liquidity_pass']:
        df[c]=df[c].fillna(False).astype(bool)
    df['session_index']=df.date.map({d:i for i,d in enumerate(sorted(df.date.unique()))})
    benchmark=df.loc[df.base].groupby('date').net20_50.mean()
    df['excess20_same_date']=df.net20_50-df.date.map(benchmark)
    result={'scope':'SUPPLEMENT_ONLY_NO_SELECTION_CHANGES','matched_date_results':{},'champion_stage_diagnostics':{}}
    for phase in ('RESEARCH','EVALUATION'):
        part=df.loc[df.phase==phase]
        result['matched_date_results'][phase]={}
        for name,mask in [('CHAMPION',part.champion),('PREBREAK',part.prebreak),('RELAUNCH',part.relaunch)]:
            selected=events(part,part.base & mask)
            result['matched_date_results'][phase][name]={
                'events':len(selected),'same_date_excess_net20':distributions(selected.excess20_same_date),
                'ticker_ci':cluster_ci(selected,'excess20_same_date'),
                'date_ci_sensitivity':cluster_ci(selected,'excess20_same_date','date')}
        raw=part.loc[part.history & part.champion]
        known=raw.loc[raw.base]
        stage=Counter()
        q=[]
        retained=[]
        removed=defaultdict(list)
        for r in known.itertuples():
            det=detail[(r.ticker,r.date)]
            fails=[]
            for k in ('swing','rsi','weinstein','quality'):
                status=det['stages'][k]['status']
                stage[k+'_'+status]+=1
                if status=='FAIL':
                    fails.append(k)
                    removed[k].append(r.Index)
            q.append(det['quality_score'])
            if not fails:
                retained.append(r.Index)
        result['champion_stage_diagnostics'][phase]={
            'raw_triggers_with_history':len(raw),'known_price_or_liquidity_fail':len(raw)-len(known),
            'price_liquidity_pass':len(known),'stage_counts_nonexclusive':dict(stage),
            'quality_default_only':distributions(pd.Series(q,dtype=float)),
            'quality_default_below78_NOT_VERIFIED_REJECTION':sum(v<78 for v in q),
            'known_downstream_clear_but_fundamentals_unknown':stats(df.loc[retained]),
            'gate_failed_all_outcomes':{k:stats(df.loc[ix]) for k,ix in removed.items()}}
    strong=events(df,df.base & (df.r20>=10))
    result['opportunity_quality_default_only']=distributions(pd.Series([detail[(r.ticker,r.date)]['quality_score'] for r in strong.itertuples()]))
    result['category2_trigger_known_price_liquidity_fail_strong_events']=len(events(df,df.history & ~df.base & df.champion & (df.r20>=10)))
    result['category6_insufficient_history_strong_events']=len(events(df,~df.history & (df.r20>=10)))
    result['predicate_failures_by_outcome_group']={}
    for horizon in (5,10,20,30):
        for threshold in (10,20,30):
            sample=events(df,df.base & (df[f'r{horizon}']>=threshold))
            result['predicate_failures_by_outcome_group'][f'close{horizon}_ge{threshold}']={
                'events':len(sample), 'failures_nonexclusive':{p:int((~sample[p].astype(bool)).sum())
                    for p in VCE_PREDICATES+RVOL_PREDICATES}}
    result['signal_density_daily']=df.loc[df.base].groupby('date')[['champion','prebreak','relaunch']].sum().astype(int).to_dict('index')
    write_json(run/'supplement_final.json',result)
    print(json.dumps({'supplement':str(run/'supplement_final.json')}))


def archive_sources(run):
    spec=json.loads((run/'spec.json').read_text())
    root=Path(__file__).resolve().parents[3]
    for name,expected in spec['sources'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('INVALID_SOURCE_PATH')
        target=run/'source'/name
        raw=target.read_bytes() if target.exists() else (root/name).read_bytes()
        if digest(raw)!=expected:
            raise ValueError('SOURCE_CHANGED_SINCE_RUN:'+name)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            with target.open('xb') as handle:
                handle.write(raw)


def seal(run):
    archive_sources(run)
    # Postprocessing may fix serialization/diagnostics without changing selections.
    current=Path(__file__).parents[1]/'research'/'analysis.py'
    for p in (current,Path(__file__)):
        target=run/'postprocess_source'/p.name
        target.parent.mkdir(exist_ok=True)
        with target.open('xb') as handle:
            handle.write(p.read_bytes())
    excluded=['summary.json'] if (run/'summary_final.json').exists() else []
    files={str(p.relative_to(run)).replace('\\','/'):digest(p.read_bytes()) for p in run.rglob('*')
           if p.is_file() and str(p.relative_to(run)).replace('\\','/') not in excluded}
    write_json(run/'manifest.json',{'complete':True,'sha256':files,'excluded_incomplete_artifacts':excluded})
    print(json.dumps({'sealed_files':len(files)}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['supplement','seal','archive'])
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    if not (args.run/'complete.json').exists():
        raise ValueError('INCOMPLETE_RUN')
    if args.command=='supplement':
        supplement(args.run)
    elif args.command=='seal':
        seal(args.run)
    else:
        archive_sources(args.run)


if __name__=='__main__':
    main()
