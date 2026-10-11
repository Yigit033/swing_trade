import copy
from datetime import date
import numpy as np
import pandas as pd
import pytest
from ..bars import normalize, MODE
from ..calendar import sessions
from .features import features, outcomes, decision_frame
from .panel import independent_indices
from .analysis import events, cluster_ci, difference_ci
from swing_trader.small_cap.signals import SmallCapSignals
from swing_trader.small_cap.settings_config import SmallCapSettings


def fixture(seed=3):
    idx=pd.DatetimeIndex(sessions(date(2025,9,4),date(2026,10,9)))
    rng=np.random.default_rng(seed)
    c=20*np.exp(np.cumsum(rng.normal(.001,.02,len(idx))))
    v=rng.lognormal(14.5,.8,len(idx))
    df=pd.DataFrame({'Open':c*.995,'High':c*1.02,'Low':c*.98,'Close':c,
                     'Volume':v,'Adj Close':c,'Dividends':0.,'Stock Splits':0.},index=idx)
    rows,_=normalize(df,idx[-1].date())
    return {'rows':rows,'mode':MODE},idx


@pytest.mark.parametrize('seed',[1,2,3,17,21])
def test_vector_predicates_match_production(seed):
    obj,idx=fixture(seed)
    settings=SmallCapSettings()
    x,_=features(obj,idx,settings)
    sig=SmallCapSignals(settings=settings)
    for t in range(130,len(idx),3):
        df=decision_frame(obj,idx[t].date())
        assert bool(x.vce.iloc[t])==sig.check_vce_breakout(df)[0]
        assert bool(x.rvol.iloc[t])==sig.check_rvol_thrust(df)[0]


def test_future_prices_do_not_change_selection():
    obj,idx=fixture()
    other=copy.deepcopy(obj)
    for r in other['rows'][180:]:
        for k in ('Open','High','Low','Close','Adj Close'):
            r[k]*=10
    a,_=features(obj,idx,SmallCapSettings())
    b,_=features(other,idx,SmallCapSettings())
    pd.testing.assert_frame_equal(a.iloc[:180],b.iloc[:180])


def test_split_reconstruction_price_and_dollar_volume():
    obj,idx=fixture()
    original=copy.deepcopy(obj)
    # A future two-for-one split restates provider history before decision.
    for row in obj['rows']:
        for k in ('Open','High','Low','Close','Adj Close'):
            row[k]/=2
        row['Volume']*=2
    obj['rows'][200]['Stock Splits']=2
    a=decision_frame(original,idx[170].date())
    b=decision_frame(obj,idx[170].date())
    pd.testing.assert_frame_equal(a,b)
    x,_=features(obj,idx,SmallCapSettings())
    y,_=features(original,idx,SmallCapSettings())
    assert x.price.iloc[170]==y.price.iloc[170]
    assert x.dv.iloc[170]==pytest.approx(y.dv.iloc[170])


def test_future_dividend_factor_cancels_at_decision():
    obj,idx=fixture()
    other=copy.deepcopy(obj)
    for r in other['rows'][:200]:
        r['Adj Close']*=.9
    a=decision_frame(obj,idx[170].date())
    b=decision_frame(other,idx[170].date())
    np.testing.assert_allclose(a.Close,b.Close)


def test_next_open_costs_and_missing_day_not_skipped():
    obj,idx=fixture()
    _,d=features(obj,idx,SmallCapSettings())
    out=outcomes(d)
    expected=(d.Close.iloc[105]/d.Open.iloc[101]-1)*100
    assert out.r5.iloc[100]==pytest.approx(expected)
    assert out.net5_50.iloc[100]<out.r5.iloc[100]
    d.loc[idx[101],:]=np.nan
    assert not outcomes(d).valid5.iloc[100]
    assert pd.isna(outcomes(d).r5.iloc[100])


def test_clusters_do_not_extend_indefinitely():
    frame=pd.DataFrame({'ticker':'AAA','session_index':range(90)})
    mask=pd.Series(True,index=frame.index)
    assert independent_indices(frame,mask)==[0,31,62]
    assert list(events(frame,mask).session_index)==[0,31,62]


def test_cluster_bootstrap_deterministic_and_paired_difference():
    frame=pd.DataFrame({'ticker':['A','A','B','C'],'r':[1.,2.,-3.,4.]})
    assert cluster_ci(frame,'r')==cluster_ci(frame,'r')
    assert difference_ci(frame,frame,'r')==[0.,0.]


def test_purge_outcome_windows_do_not_overlap_evaluation():
    # Research decisions0..29 have outcomes through59; evaluation starts60.
    assert max(range(30))+30<min(range(60,90))


def test_protocol_has_exactly_two_fixed_hypotheses():
    from .protocol import PROTOCOL
    assert len([k for k in PROTOCOL if k.startswith('challenger_')])==2
    assert PROTOCOL['no_optimization'] is True
