"""Conservative offline same-session legacy adapter; NEVER fetches Finviz/DB."""
import argparse
import json
from pathlib import Path
from ..legacy import recorded_run_legacy
from ..master import write_json
from swing_trader.small_cap.settings_config import SmallCapSettings, _prune_removed_keys


def adapt(run, session, settings):
    result=recorded_run_legacy(run)
    st=run['stats']
    cfg=settings.universe_scan
    ranked=st.get('universe_ranked_total')
    members=st.get('scanned_members',[])
    reasons=[]
    if st.get('bar_session')!=session:
        reasons.append('NOT_SAME_SESSION')
    if not cfg.use_finviz or st.get('universe_fallback'):
        reasons.append('STATIC_OR_DISABLED')
    if ranked is None or min(ranked,cfg.max_scan_tickers)<cfg.min_finviz_tickers_skip_static_merge:
        reasons.append('STATIC_MERGE_NOT_EXCLUDED')
    if ranked is not None and st.get('stocks_scanned')!=min(ranked,cfg.max_scan_tickers):
        reasons.append('RANK_MEMBERSHIP_COUNT_MISMATCH')
    if not members or any(m.get('date')!=session for m in members):
        reasons.append('MEMBER_SESSION_UNKNOWN_OR_MIXED')
    if not result['complete']:
        reasons.append('INCOMPLETE_MEMBERSHIP')
    result['complete']=not reasons
    result['finviz_membership_inferred_from_rank_telemetry']=not reasons
    result['phase3_validation_reasons']=reasons
    result['provenance_warning']='INFERRED_FROM_RECORDED_SELECTION_NOT_FOUR_RAW_QUERY_RESPONSES'
    if reasons:
        result['finviz_selected']=[]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--run-id',type=int,required=True)
    p.add_argument('--session',required=True)
    p.add_argument('--settings',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    runs=json.loads(args.input.read_text(encoding='utf-8'))
    found=[r for r in runs if r['id']==args.run_id]
    if len(found)!=1:
        raise ValueError('RUN_ID_NOT_UNIQUE')
    settings=SmallCapSettings.model_validate(_prune_removed_keys(json.loads(args.settings.read_text(encoding='utf-8'))))
    result=adapt(found[0],args.session,settings)
    write_json(args.output,result)
    print(json.dumps({'complete':result['complete'],'reasons':result['phase3_validation_reasons'],
                      'members':len(result['finviz_selected'])}))


if __name__=='__main__':
    main()
