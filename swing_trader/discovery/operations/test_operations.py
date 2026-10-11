import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import pytest
from unittest.mock import patch
from ..bars import object_id, store_object
from ..master import write_json, digest
from ..research.test_research import fixture
from ..shadow import save_scan
from swing_trader.small_cap.settings_config import SmallCapSettings
from .__main__ import observe


def setup_files(tmp_path):
    obj,idx=fixture()
    obj['fetched_at']='2026-10-10T12:00:00+00:00'
    key=store_object(tmp_path/'cache',obj)
    scan={'inputs':{'session':'2026-10-09','decision_at':'2026-10-10T13:00:00+00:00',
                   'universe_id':'test','code':{}},'scan_id':'test',
          'rows':[{'ticker':'AAA','legacy_membership':'UNKNOWN','trigger_positive':False,'stages':{}}]}
    inputs={'settings':SmallCapSettings().model_dump(mode='json'),
            'preparation':{'outcomes':[{'symbol':'AAA','object_id':key}]}}
    save_scan(tmp_path/'scan',scan,inputs)
    return tmp_path/'scan',tmp_path/'cache'


def fake_features(*args):
    return pd.DataFrame([{'base':True,'prebreak':True,'relaunch':False,'price':20.,'atr_pct':3.,'dv':1e7}]),None


def test_prospective_separate_deduplicated_and_unknown_legacy(tmp_path):
    scan,cache=setup_files(tmp_path)
    now=datetime.fromisoformat('2026-10-11T12:00:00+00:00')
    with patch('swing_trader.discovery.operations.__main__.features',fake_features):
        book=observe(scan,cache,now=now)
        again=observe(scan,cache,previous=book,now=now)
    assert len(book['observations'])==len(again['observations'])==1
    row=book['observations'][0]
    assert row['entry_session']=='2026-10-12'
    assert not row['broad_only_confirmed']
    assert row['future_status']=='PENDING'


def test_reject_append_to_champion_book(tmp_path):
    scan,cache=setup_files(tmp_path)
    with pytest.raises(ValueError,match='CHAMPION_BOOK'):
        observe(scan,cache,previous={'observations':[]},now=datetime.fromisoformat('2026-10-11T12:00:00+00:00'))


def test_reject_backdated_challenger_registration(tmp_path):
    scan,cache=setup_files(tmp_path)
    with pytest.raises(ValueError,match='PRE_ENTRY'):
        observe(scan,cache,now=datetime.fromisoformat('2026-10-12T14:00:00+00:00'))


def test_modified_scan_fails_integrity(tmp_path):
    scan,cache=setup_files(tmp_path)
    (scan/'scan.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='MANIFEST'):
        observe(scan,cache,now=datetime.fromisoformat('2026-10-11T12:00:00+00:00'))


def test_historical_panel_reproducible_same_inputs(tmp_path):
    from ..research.panel import run
    from ..research.protocol import PROTOCOL
    obj,idx=fixture()
    outcomes=[]
    for symbol in ('AAA','SPY','^VIX'):
        value={**obj,'symbol':symbol,'vendor_symbol':symbol,
               'identity':{'symbol':symbol,'exchangeName':'NMS','instrumentType':'EQUITY','currency':'USD'}}
        outcomes.append({'symbol':symbol,'object_id':store_object(tmp_path/'cache',value)})
    prep={'session':'2026-10-09','outcomes':outcomes}
    inventory={'protocol':PROTOCOL,'preparation_id':object_id(prep),
               'chosen_sessions':[str(d.date()) for d in idx[-120:-30]],'first_dates':{str(idx[0].date()):1}}
    inputs={'settings':SmallCapSettings().model_dump(mode='json'),
            'snapshot':{'records':[{'symbol':'AAA','exchange':'NASDAQ','discovery_status':'CANDIDATE',
                                   'discovery_reason':'equity','security_type':'COMMON_STOCK'}]}}
    run(inputs,prep,tmp_path/'cache',inventory,tmp_path/'run1')
    run(inputs,prep,tmp_path/'cache',inventory,tmp_path/'run2')
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path/'run1/panels/00000.csv.gz'),
                                  pd.read_csv(tmp_path/'run2/panels/00000.csv.gz'))
    a=json.loads((tmp_path/'run1/details/00000.json').read_text())
    b=json.loads((tmp_path/'run2/details/00000.json').read_text())
    assert a['details']==b['details']
    assert a['parity_checks']==b['parity_checks']


def test_legacy_static_merge_is_not_finviz_membership():
    from .legacy import adapt
    settings=SmallCapSettings()
    run={'id':1,'created_at':'2026-10-10T00:00:00Z','stats':{'bar_session':'2026-10-09',
         'stocks_scanned':2,'universe_ranked_total':1,'scanned_members':[
             {'ticker':'AAA','date':'2026-10-09'},{'ticker':'BBB','date':'2026-10-09'}]}}
    result=adapt(run,'2026-10-09',settings)
    assert not result['complete']
    assert result['finviz_selected']==[]
    assert 'STATIC_MERGE_NOT_EXCLUDED' in result['phase3_validation_reasons']


def test_legacy_same_session_and_complete_members_required():
    from .legacy import adapt
    settings=SmallCapSettings()
    run={'id':1,'created_at':'2026-10-10T00:00:00Z','stats':{'bar_session':'2026-10-09',
         'stocks_scanned':30,'universe_ranked_total':30,'scanned_members':[
             {'ticker':f'T{i}','date':'2026-10-09'} for i in range(30)]}}
    assert adapt(run,'2026-10-09',settings)['complete']
    assert not adapt(run,'2026-10-08',settings)['complete']


def test_nonfinite_diagnostic_distribution_is_counted_not_serialized():
    from ..research.analysis import distributions
    result=distributions(pd.Series([1.,2.,float('inf'),float('nan')]))
    assert result['mean']==1.5
    assert result['nonfinite_or_missing']==2
    json.dumps(result,allow_nan=False)
