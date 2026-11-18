"""Offline source-cluster inference; no API calls, no automatic data access.

Counts are ordered (S,A)=00,01,10,11. Missing measurements stay in the
assigned denominator. Bootstrap envelopes are descriptive approximations,
not distribution-free guarantees or evidence of autonomous agent behavior.
"""
from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
import math
import json
from .calibration_robustness import tv_extreme


def row_options(rows):
    if not rows or len({x['id'] for x in rows})!=len(rows):raise ValueError('Unique nonempty response IDs required')
    result=[]
    for x in rows:
        if not x.get('task') or not x.get('source_hash'):raise ValueError('Source and task required')
        for k in ('s','a'):
            if k not in x or (x[k] is not None and (type(x[k]) is not int or x[k] not in (0,1))):raise ValueError('Observed binary value or explicit None required')
        result.append(tuple(2*s+a for s in ((0,1) if x['s'] is None else (x['s'],))
                            for a in ((0,1) if x['a'] is None else (x['a'],))))
    return result


@lru_cache(maxsize=256)
def _count_offsets(weighted_options):
    """Exact attainable counts, merging equivalent partial assignments.

    Each term is one original row with its cluster-resampling multiplicity.
    All copies of that row must take the same cell. Inductively the state set
    contains every attainable count vector, so merging duplicates loses no
    extremum of any statistic depending only on the four joint counts.
    """
    worlds={(0,0,0,0)}
    for weight,cells in weighted_options:
        if weight==0:continue
        following=set()
        for counts in worlds:
            for cell in cells:
                value=list(counts);value[cell]+=weight;following.add(tuple(value))
        if len(following)>1000000:
            raise ValueError('Too many distinct count states; no missing cases removed')
        worlds=following
    return tuple(sorted(worlds))


def completion_counts(rows):
    options=row_options(rows);base=[0]*4;missing=[]
    for cells in options:
        if len(cells)==1:base[cells[0]]+=1
        else:missing.append(cells)
    return [tuple(b+v for b,v in zip(base,offset))
            for offset in _count_offsets(tuple(sorted((1,cells) for cells in missing)))]


def law_metrics(counts,R=F(22),D=F(7757,1050),c=F(1)):
    if len(counts)!=4 or any(type(x) is not int or x<0 for x in counts) or sum(counts)==0:raise ValueError('Four nonnegative integer counts required')
    q00,q01,q10,q11=[F(v,sum(counts)) for v in counts];h=q10+q11;p=q01+q11;delta=q11-h*p
    target=q11*R-q10*D-c;blind=p*R-(1-p)*D;inverse=q01*R-q00*D-c
    return dict(h=float(h),a0=float(p),a_plus=float(q11/h) if h else None,
        a_minus=float(q01/(1-h)) if h<1 else None,delta=float(delta),
        weak_min_reward=float(c*(1-p)/delta) if delta>0 else float('inf'),
        target_credit=float(target),blind_credit=float(blind),
        target_menu_margin=float(target-max(F(0),blind,-c,inverse,blind-c)))


def calibrate(count_worlds,radius=F(0),c=F(1),blind_margin=F(1,100),target_margin=F(1,20)):
    """Exact missing/TV-robust contract, minimizing reward then rounding up.

    With c>0, target>0 and blind<0 also beat the other four menu policies.
    Let p* be worst blind acceptance and (v10,v11) a worst target law.
    Eliminating D gives R[v11(1-p*)-v10*p*] >= (c+t)(1-p*)+v10*b.
    The TV rankings do not depend on R,D>=0; this elimination is exact.
    TV radius is a sensitivity parameter, not an estimated confidence radius.
    """
    c,blind_margin,target_margin,radius=map(F,(c,blind_margin,target_margin,radius))
    if not count_worlds or c<=0 or min(blind_margin,target_margin)<0:raise ValueError('Invalid calibration inputs')
    for counts in count_worlds:law_metrics(counts)
    laws=[tuple(F(v,sum(counts)) for v in counts) for counts in count_worlds]
    blind=[tv_extreme(q,(-1,1,-1,1),radius) for q in laws]
    targets=[tv_extreme(q,(0,0,-1,1),radius,maximize=False) for q in laws]
    p=max(q[1]+q[3] for q in blind)
    if p>=1:return None
    lower=[]
    for q in targets:
        denom=q[3]*(1-p)-q[2]*p
        if denom<=0:return None
        lower.append(((c+target_margin)*(1-p)+q[2]*blind_margin)/denom)
    minimum=max(lower);R=F(math.ceil(minimum));lo=(p*R+blind_margin)/(1-p)
    upper=[(q[3]*R-c-target_margin)/q[2] for q in targets if q[2]>0]
    hi=min(upper) if upper else None;D=(lo+hi)/2 if hi is not None else lo
    if hi is not None and hi<lo:raise ArithmeticError('Elimination inconsistency')
    if any(q[3]*R-q[2]*D-c<target_margin for q in targets):raise ArithmeticError('Target constraint violated')
    return dict(minimum_reward=str(minimum),R=str(R),D=str(D),deposit_interval=[str(lo),str(hi) if hi is not None else None],
                radius=str(radius),c=str(c),blind_margin=str(blind_margin),target_margin=str(target_margin))


def policy_bounds(rows,name,R=F(22),D=F(7757,1050),c=F(1)):
    if name not in ('idle','blind','M:00','M:01','M:10','M:11'):raise ValueError('Unknown policy')
    options=row_options(rows);lo=defaultdict(lambda:F(0));hi=defaultdict(lambda:F(0));unknown=False
    for row,cells in zip(rows,options):
        if type(row.get('y')) is not int or row['y'] not in (0,1):raise ValueError('Binary reference required')
        values=[]
        for cell in cells:
            s,a=divmod(cell,2);f=int(name=='blind') if name in ('idle','blind') else int(name[2] if s else name[3]);y=row['y'];uses_m=name.startswith('M:')
            mt,at=row.get('m_tokens'),row.get('a_tokens')
            if any(v is not None and (type(v) is not int or v<0) for v in (mt,at)):raise ValueError('Unknown or nonnegative token usage required')
            unknown|=(uses_m and mt is None) or (bool(f) and at is None)
            values.append(dict(queries=f,accepted=f*a,rejected=f*(1-a),fn=y*(1-f*a),fp=(1-y)*f*a,
                lost_acceptances=a*(1-f),avoided_rejections=(1-a)*(1-f),
                credit=f*(a*R-(1-a)*D)-int(uses_m)*c,
                known_conditional_tokens=int(uses_m)*(mt or 0)+f*(at or 0)))
        for key in values[0]:lo[key]+=min(x[key] for x in values);hi[key]+=max(x[key] for x in values)
    out={k:[int(lo[k]),int(hi[k])] for k in lo if k not in ('credit','known_conditional_tokens')}
    out['mean_credit']=[float(lo['credit']/len(rows)),float(hi['credit']/len(rows))]
    out['conditional_tokens']=dict(bounds=[int(lo['known_conditional_tokens']),None if unknown else int(hi['known_conditional_tokens'])],unknown_possible=unknown,
        interpretation='Potential workflow cost,not actual paid saving;unknown usage has no finite certified upper bound.')
    return out


def metadata_only_bounds(rows):
    row_options(rows);fn=[0,0];fp=[0,0];known=0;unknown=0
    for row in rows:
        if type(row.get('y')) is not int or row['y'] not in (0,1):raise ValueError('Binary reference required')
        choices=(0,1) if row['s'] is None else (row['s'],)
        misses=[row['y']*(1-s0) for s0 in choices]
        false=[(1-row['y'])*s0 for s0 in choices]
        fn[0]+=min(misses);fn[1]+=max(misses);fp[0]+=min(false);fp[1]+=max(false)
        token=row.get('m_tokens')
        if token is None:unknown+=1
        elif type(token) is int and token>=0:known+=token
        else:raise ValueError('Invalid token usage')
    return dict(fn=fn,fp=fp,screenings=len(rows),adjudications=0,conditional_tokens=[known,None if unknown else known],
        interpretation='M-only quality baseline without final A;not a challenge-contract policy and no credit payoff assigned.')


def _array_metrics(q,R,D,c):
    import numpy as np
    q00,q01,q10,q11=np.moveaxis(q,-1,0);h=q10+q11;p=q01+q11;delta=q11-h*p
    target=q11*R-q10*D-c;blind=p*R-(1-p)*D;inverse=q01*R-q00*D-c
    with np.errstate(divide='ignore',invalid='ignore'):
        return dict(h=h,a0=p,a_plus=np.where(h>0,q11/h,np.nan),a_minus=np.where(h<1,q01/(1-h),np.nan),delta=delta,
            weak_min_reward=np.where(delta>1e-14,c*(1-p)/delta,np.inf),target_credit=target,blind_credit=blind,
            target_menu_margin=target-np.maximum.reduce([np.zeros_like(h),blind,np.full_like(h,-c),inverse,blind-c]))


def bootstrap(rows,replicates=10000,seed=20260917087,R=F(22),D=F(7757,1050),c=F(1)):
    """Task-stratified source resampling with all missing-state completions.

    Percentile endpoints enclose lower/upper identification-bound estimates.
    Undefined conditional probabilities are not silently set to zero; if any
    draw admits an undefined value, that metric's interval is left unavailable.
    """
    import numpy as np
    if type(replicates) is not int or replicates<1:raise ValueError('Positive resample count required')
    options=row_options(rows);keys=sorted({(x['task'],x['source_hash']) for x in rows});index={k:i for i,k in enumerate(keys)}
    base=np.zeros((len(keys),4),dtype=np.int64);sizes=np.zeros(len(keys),dtype=np.int64);missing=[]
    for row,cells in zip(rows,options):
        j=index[row['task'],row['source_hash']];sizes[j]+=1
        if len(cells)==1:base[j,cells[0]]+=1
        else:missing.append((j,cells))
    tasks=defaultdict(list)
    for j,(task,_) in enumerate(keys):tasks[task].append(j)
    rng=np.random.default_rng(seed);lowers=defaultdict(list);uppers=defaultdict(list);undefined=defaultdict(int)
    for start in range(0,replicates,250):
        n=min(250,replicates-start);weights=np.zeros((n,len(keys)),dtype=np.int64)
        for indices in tasks.values():weights[:,indices]=rng.multinomial(len(indices),np.full(len(indices),1/len(indices)),size=n)
        counts=weights@base;denom=weights@sizes
        if not missing:
            metrics=_array_metrics(counts/denom[:,None],float(R),float(D),float(c))
            for key,value in metrics.items():
                lowers[key].extend(value);uppers[key].extend(value);undefined[key]+=int(np.isnan(value).sum())
        else:
            for i in range(n):
                terms=tuple(sorted((int(weights[i,j]),cells) for j,cells in missing if weights[i,j]))
                offsets=np.asarray(_count_offsets(terms),dtype=np.int64)
                metrics=_array_metrics((counts[i]+offsets)/denom[i],float(R),float(D),float(c))
                for key,value in metrics.items():
                    valid=value[~np.isnan(value)]
                    lowers[key].append(float(valid.min()) if len(valid) else float('nan'))
                    uppers[key].append(float(valid.max()) if len(valid) else float('nan'))
                    undefined[key]+=int(len(valid)!=len(value))
    result={}
    for key in lowers:
        interval=None if undefined[key] else [float(np.quantile(lowers[key],.025,method='inverted_cdf')),float(np.quantile(uppers[key],.975,method='inverted_cdf'))]
        result[key]=dict(interval=interval,undefined_draws=undefined[key],infinite_upper_draws=sum(math.isinf(float(x)) for x in uppers[key]))
    return dict(responses=len(rows),sources=len(keys),replicates=replicates,seed=seed,missing_rows=len(missing),metrics=result,
        interpretation='Task-stratified source-cluster percentile envelope,conditional empirical approximation;not exact coverage,not iid responses,not causal agent behavior.')
