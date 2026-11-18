"""Hand cases catch wrong TV convention, missing-cell deletion and IC signs."""
import importlib
from fractions import Fraction as F
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def mod():
    assert importlib.util.find_spec('experiments.calibration_robustness') is not None, 'Calibration uncertainty analysis missing'
    return importlib.import_module('experiments.calibration_robustness')


def centres():
    return [(F(1,4), F(0), F(21,40), F(9,40)),
            (F(1,4), F(0), F(1,2), F(1,4))]


def test_unknown_adjudication_is_completed_in_both_worlds_not_dropped():
    rows=[dict(id='x',m=.9,a=None),dict(id='y',m=.1,a=1),dict(id='z',m=.2,a=0)]
    assert mod().completed_counts(rows)==[(1,1,1,0),(1,1,0,1)]


@pytest.mark.parametrize('rows', [[],[dict(id='x',m=.8,a=2)],
    [dict(id='x',m=float('nan'),a=0)],
    [dict(id='x',m=.5,a=0),dict(id='x',m=.2,a=1)]])
def test_bad_or_duplicate_rows_do_not_become_calibration(rows):
    with pytest.raises(ValueError):mod().completed_counts(rows)


def test_simultaneous_box_uses_four_cells_and_missing_completion_envelope():
    box=mod().confidence_box([(2,0,1,1),(2,0,0,2)],alpha=.05)
    assert box[1]==pytest.approx((0,1-(.05/8)**.25))
    assert box[2][0]==0
    assert box[2][1]>0
    assert box[3][0]<.25 and box[3][1]>.5
    with pytest.raises(ValueError):mod().confidence_box([(1,0,0,0),(2,0,0,0)])


def test_tv_extremizer_moves_mass_not_twice_tv_and_obeys_simplex():
    m=mod();q=(F(1,2),F(0),F(1,2),F(0));v=(0,4,-2,4)
    w=m.tv_extreme(q,v,F(1,4))
    assert sum(w)==1 and min(w)>=0
    assert sum(x*y for x,y in zip(w,v))==F(1,2)
    assert sum(abs(x-y) for x,y in zip(q,w))/2==F(1,4)
    w=m.tv_extreme(q,v,F(1,4),maximize=False)
    assert sum(x*y for x,y in zip(w,v))==F(-3,2)
    assert sum(x*y for x,y in zip(m.tv_extreme(q,v,F(1)),v))==4
    with pytest.raises(ValueError):m.tv_extreme(q,v,F(-1,10))


def test_old_point_contract_recovered_with_original_safety_margins():
    m=mod()
    assert m.minimum_reward(centres(),F(0),c=F(1),bmargin=F(1,100),tmargin=F(1,20))==(F(1057,50),F(353,50))
    assert m.deposit_interval(centres(),F(0),F(22),c=F(1),bmargin=F(1,100),tmargin=F(1,20))==(F(551,75),F(52,7))


def test_robust_interval_rejects_one_cell_flip_scale_for_every_reward():
    m=mod()
    assert m.minimum_reward(centres(),F(1,1000))==(F(2996,143),F(1004,143))
    assert m.minimum_reward(centres(),F(1,40)) is None
    assert m.deposit_interval(centres(),F(1,500),F(22))==(F(126,17),F(126,17))
    assert m.deposit_interval(centres(),F(1,500)+F(1,1000000),F(22)) is None


def test_same_reward_range_certificate_balances_both_incentive_constraints():
    result=mod().range_certificate(centres(),F(22))
    assert result==dict(R=F(22),D=F(126,17),radius=F(1,500))
    assert mod().range_certificate(centres(),F(44))==dict(R=F(44),D=F(796,51),radius=F(9,760))
    assert mod().range_certificate(centres(),F(19)) is None


def test_independence_witness_has_correct_marginals_and_no_information_rent():
    m=mod();q=m.independence(centres()[0])
    assert q==(F(31,160),F(9,160),F(93,160),F(27,160))
    assert m.minimum_reward([q],F(0)) is None
    box=[(0.,1.)]*4
    assert m.inside_box(q,box)
    assert not m.inside_box(q,[(0.,.1)]+box[1:])
