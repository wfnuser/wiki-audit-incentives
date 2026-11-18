"""r84: old-calibration-only uncertainty and contract sensitivity; no API/writes.

Order: (S,A) = 00,01,10,11. TV is half the L1 distance. A confidence
region needs iid, fixed-protocol source/output pairs; this historical,
adaptively used calibration does not itself establish that assumption.
"""
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import hashlib
import json
import math

from .incentive_uncertainty import cp_interval


def completed_counts(rows):
    if not rows or any(not isinstance(r.get('id'),str) or not r['id'] for r in rows):
        raise ValueError('Nonempty identified rows required')
    if len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Duplicate calibration source ID')
    for r in rows:
        if type(r.get('m')) not in (int,float) or not math.isfinite(r['m']) or not 0<=r['m']<=1:
            raise ValueError('Finite signal in [0,1] required')
        if r.get('a') is not None and (type(r['a']) is not int or r['a'] not in (0,1)):
            raise ValueError('Binary or explicitly missing adjudication required')
        if 'a' not in r:raise ValueError('Missing adjudication key')
    missing=sum(r['a'] is None for r in rows)
    if missing>1:raise ValueError('This audit supports at most one missing adjudication')
    result=[]
    for endpoint in range(2 if missing else 1):
        counts=[0]*4
        for r in rows:counts[2*int(r['m']>=.5)+(endpoint if r['a'] is None else r['a'])]+=1
        result.append(tuple(counts))
    return result


def confidence_box(completions,alpha=.05):
    """Four central CP intervals, each alpha/4; envelope of missing completions.

    The envelope contains the interval computed with the unknown true
    completion. There is no imputation-at-random assumption. Intersect the
    returned box with the probability simplex, rather than treating its
    four intervals as four independent random parameters.
    """
    if not completions or not 0<alpha<1:
        raise ValueError('Nonempty completions and alpha in (0,1) required')
    for counts in completions:
        if len(counts)!=4 or any(type(k) is not int or k<0 for k in counts) or sum(counts)<1:
            raise ValueError('Four integer nonnegative counts with positive total required')
    n=sum(completions[0])
    if any(sum(v)!=n for v in completions):raise ValueError('Completions need identical denominators')
    intervals=[[cp_interval(k,n,alpha/4) for k in row] for row in completions]
    return [(min(row[j][0] for row in intervals),max(row[j][1] for row in intervals)) for j in range(4)]


def _exact(x):
    if type(x) is int or isinstance(x,F):return F(x)
    raise ValueError('Integer/Fraction required for exact contract arithmetic')


def _law(q):
    q=tuple(map(_exact,q))
    if len(q)!=4 or min(q)<0 or sum(q)!=1:raise ValueError('Four-cell probability law required')
    return q


def independence(q):
    q=_law(q);h=q[2]+q[3];p=q[1]+q[3]
    return ((1-h)*(1-p),(1-h)*p,h*(1-p),h*p)


def inside_box(q,box):
    q=_law(q)
    if len(box)!=4 or any(not 0<=lo<=hi<=1 for lo,hi in box):raise ValueError('Invalid probability box')
    return all(lo<=float(v)<=hi for v,(lo,hi) in zip(q,box))


def tv_extreme(q,values,radius,maximize=True):
    """Exact linear extremizer on simplex intersected with a TV ball.

    Move mass from lowest to highest coefficient (reverse for min). Stop
    at equal coefficients or simplex boundaries, so the range bound is
    not incorrectly extrapolated past available donor mass.
    """
    q=_law(q);values=tuple(map(_exact,values));radius=_exact(radius)
    if len(values)!=4 or not 0<=radius<=1:raise ValueError('Four values and TV radius in [0,1] required')
    values=values if maximize else tuple(-v for v in values)
    order=sorted(range(4),key=lambda i:values[i]);w=list(q);lo=0;hi=3;remaining=radius
    while remaining and lo<hi:
        i,j=order[lo],order[hi]
        if values[i]==values[j]:break
        amount=min(remaining,w[i],1-w[j])
        w[i]-=amount;w[j]+=amount;remaining-=amount
        if w[i]==0:lo+=1
        if w[j]==1:hi-=1
    return tuple(w)


def _constraints(centres,radius,c,bmargin,tmargin):
    c,bmargin,tmargin=map(_exact,(c,bmargin,tmargin))
    if not centres or c<=0 or min(bmargin,tmargin)<0:raise ValueError('Nonempty laws, positive cost and nonnegative margins required')
    cs=[(F(1),F(0),F(0)),(F(0),F(1),F(0))]
    for q in centres:
        # The rankings are identical for every R,D>=0, even with ties.
        wb=tv_extreme(q,(-1,1,-1,1),radius)
        wt=tv_extreme(q,(0,0,-1,1),radius,maximize=False)
        p=wb[1]+wb[3]
        cs.extend([(-p,1-p,bmargin),(wt[3],-wt[2],c+tmargin)])
    return cs


def deposit_interval(centres,radius,R,c=F(1),bmargin=F(0),tmargin=F(0)):
    R=_exact(R)
    if R<0:raise ValueError('Nonnegative reward required')
    lo=F(0);hi=None
    for a,b,g in _constraints(centres,radius,c,bmargin,tmargin):
        rhs=g-a*R
        if b>0:lo=max(lo,rhs/b)
        elif b<0:hi=rhs/b if hi is None else min(hi,rhs/b)
        elif rhs>0:return None
    return None if hi is not None and lo>hi else (lo,hi)


def minimum_reward(centres,radius,c=F(1),bmargin=F(0),tmargin=F(0)):
    cs=_constraints(centres,radius,c,bmargin,tmargin);feasible=[]
    for (a,b,g),(d,e,h) in combinations(cs,2):
        det=a*e-b*d
        if not det:continue
        R,D=(g*e-b*h)/det,(a*h-g*d)/det
        if all(x*R+y*D>=z for x,y,z in cs):feasible.append((R,D))
    return min(feasible) if feasible else None


def range_certificate(centres,R,c=F(1),bmargin=F(0),tmargin=F(0)):
    """Maximize sufficient min-slack/(R+D) over deposits at fixed R>0.

    This maximizes the range-based certificate, not always the exact TV
    radius after simplex saturation. The audit checks exact feasibility
    at the reported radius and just above it for its historical laws.
    """
    R=_exact(R)
    if R<=0:raise ValueError('Positive reward required')
    lines=[(b,a*R-g) for a,b,g in _constraints(centres,F(0),c,bmargin,tmargin)[2:]]
    candidates={F(0)}
    for (b,k),(d,h) in combinations(lines,2):
        if b!=d and (h-k)/(b-d)>=0:candidates.add((h-k)/(b-d))
    # A positive-cost target with q10=0 may have an asymptotic supremum;
    # its decreasing (or constant) target slack/R+D still permits a finite
    # optimum here. All candidate values are independently feasible-tested.
    options=[(min(b*D+k for b,k in lines)/(R+D),-D) for D in candidates]
    radius,minus_d=max(options)
    if radius<0:return None
    return dict(R=R,D=-minus_d,radius=radius)


def main():
    root=Path(__file__).resolve().parents[1]
    hashes={
        'acceptance_forecast_v1_summary.json':'931d9a824b2b65837306b489e7534677fd89106a2d71be8ef63e834da5695f66',
        'metadata_gate_completion_v1_summary.json':'5e8f101e082feee368fd7545a9bfdbb9a550d4f1dcc0539af045b0d97fb187cf',
    }
    rows=[];per_batch=[]
    for name,sha in hashes.items():
        raw=(root/'artifacts'/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Frozen calibration hash mismatch')
        batch=json.loads(raw)['items'];rows.extend(batch)
        per_batch.append(dict(file=name,n=len(batch),counts=completed_counts(batch)))
    if len(rows)!=40 or sum(r['a'] is None for r in rows)!=1:raise ValueError('Expected old40 with one missing A')
    counts=completed_counts(rows);centres=[tuple(F(k,40) for k in x) for x in counts]
    box=confidence_box(counts);witnesses=[independence(q) for q in centres]
    R,D,c=F(22),F(7757,1050),F(1)
    slacks=[dict(blind=-(q[1]+q[3])*R+(q[0]+q[2])*D,
                 target=q[3]*R-q[2]*D-c) for q in centres]
    radius=min(x for row in slacks for x in row.values())/(R+D)
    frontier=[]
    for reward in (22,44,88):
        out=range_certificate(centres,F(reward));eps=out['radius']
        out['exact_interval_at_radius']=deposit_interval(centres,eps,F(reward))
        out['interval_just_above']=deposit_interval(centres,eps+F(1,1000000),F(reward))
        frontier.append(out)
    result=dict(version='calibration-robustness-r84',new_api_calls=0,source_sha256=hashes,
        source_count=40,missing_a=1,per_batch=per_batch,counts=counts,
        simultaneous_box=dict(alpha=.05,per_cell_alpha=.0125,bounds=box,
            independence_witnesses=witnesses,contains=[inside_box(q,box) for q in witnesses]),
        original_contract=dict(R=R,D=D,c=c,slacks=slacks,weak_tv_radius=radius,
            exact_interval_at_radius=deposit_interval(centres,radius,R)),
        weak_tv_frontier=frontier,
        original_margin_reconstruction=dict(minimum=minimum_reward(centres,F(0),bmargin=F(1,100),tmargin=F(1,20)),
            deposit_interval=deposit_interval(centres,F(0),R,bmargin=F(1,100),tmargin=F(1,20))),
        tv_one_observation_scale=dict(radius=F(1,40),minimum_reward=minimum_reward(centres,F(1,40))),
        boundary='Old-calibration-only retrospective robustness audit. TV frontier uses weak IC, not the old .05/.01 margins. CP coverage is conditional on iid fixed protocol, not guaranteed for adaptively selected historical analyses. No new behavior, quality or savings claim; no refit of new60.')
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))


if __name__=='__main__':main()
