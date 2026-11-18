"""Read-only, post-hoc fixed-contract uncertainty audit. No API, fitting or file writes.

Input: the sealed r78 summary's source-level potential outcomes.
Binomial intervals assume iid source-output pairs and are marginal, not simultaneous.
The bootstrap distribution is enumerated exactly; its population coverage is NOT exact.
Unobserved outcome cells retain zero empirical bootstrap probability.
"""
from collections import Counter, defaultdict
from fractions import Fraction as F
from math import comb, fsum
from pathlib import Path
import hashlib
import json


def cp_interval(k, n, alpha=.05, one_sided=False):
    """Clopper-Pearson: central interval, or [0, upper] one-sided interval."""
    if type(n) is not int or type(k) is not int or not 0 <= k <= n or n < 1 or not 0 < alpha < 1:
        raise ValueError('Require integer 0<=k<=n, n>0 and 0<alpha<1')
    tail=alpha if one_sided else alpha/2

    def cdf(p, stop):
        return fsum(comb(n,j)*p**j*(1-p)**(n-j) for j in range(stop+1))

    def invert(stop, target):
        lo,hi=0.,1.
        for _ in range(80):
            mid=(lo+hi)/2
            if cdf(mid,stop)>target:lo=mid
            else:hi=mid
        return (lo+hi)/2

    lower=0. if one_sided or k==0 else invert(k-1,1-tail)
    upper=1. if k==n else invert(k,tail)
    return lower,upper


def _utilities(counts, R, D, c):
    n=sum(counts)
    q00,q01,q10,q11=(F(x,n) for x in counts)
    plus=q11*R-q10*D
    minus=q01*R-q00*D
    return {'idle':F(0),'blind':plus+minus,'M:00':-c,
            'M:01':minus-c,'M:10':plus-c,'M:11':plus+minus-c}


def _compositions(n,k):
    if k==1:
        yield (n,)
    else:
        for x in range(n+1):
            for rest in _compositions(n-x,k-1):yield (x,)+rest


def exact_bootstrap(counts,R,D,c):
    """Enumerate joint-cell bootstrap counts, using the SAME fixed contract."""
    if len(counts)!=4 or any(type(x) is not int or x<0 for x in counts) or sum(counts)<1:
        raise ValueError('Four nonnegative integer cell counts with positive total required')
    n=sum(counts);active=[i for i,x in enumerate(counts) if x]
    if comb(n+len(active)-1,len(active)-1)>200000:
        raise ValueError('Exact enumeration exceeds audit limit')
    distribution=defaultdict(F);blind_distribution=defaultdict(F)
    optimal=F(0);joint=F(0);total=F(0);cases=0
    for composition in _compositions(n,len(active)):
        weight=F(1);remaining=n;new=[0]*4
        for i,k in zip(active,composition):
            weight*=comb(remaining,k)*F(counts[i],n)**k
            remaining-=k;new[i]=k
        values=_utilities(new,R,D,c);target=values['M:10']
        unique=all(target>v for p,v in values.items() if p!='M:10')
        optimal+=weight*unique;joint+=weight*(unique and values['blind']<=0)
        distribution[target]+=weight;blind_distribution[values['blind']]+=weight
        total+=weight;cases+=1
    if total!=1:raise AssertionError('Multinomial mass is not one')

    def quantiles(dist):
        out=[]
        for threshold in [F(1,40),F(39,40)]:
            cumulative=F(0)
            for value,weight in sorted(dist.items()):
                cumulative+=weight
                if cumulative>=threshold:
                    out.append(value);break
        return tuple(out)

    return dict(mass=total,compositions=cases,
                target_mean=sum((v*w for v,w in distribution.items()),F(0)),
                target_positive_mass=sum((w for v,w in distribution.items() if v>0),F(0)),
                target_unique_best_mass=optimal,target_best_and_blind_nonpositive_mass=joint,
                target_quantiles_95=quantiles(distribution),blind_quantiles_95=quantiles(blind_distribution),
                zero_cell_caveat='Zero empirical cells stay absent. These masses are not probabilities that population IC holds.')


def analyze(items,R,D,c):
    if len(items)<2 or len({x.get('id') for x in items})!=len(items):
        raise ValueError('At least two unique source records required')
    cells=[]
    for row in items:
        if row.get('id') is None or row.get('a') not in (0,1) or row.get('y') not in (0,1):
            raise ValueError('ID and known binary outcomes required')
        if not isinstance(row.get('m'),(float,int)) or not 0<=row['m']<=1:
            raise ValueError('Known screening probability in [0,1] required')
        cells.append(2*int(row['m']>=.5)+row['a'])
    if any(not isinstance(v,(F,int)) or isinstance(v,bool) or v<0 for v in (R,D,c)):
        raise ValueError('Use nonnegative exact contract numbers')
    counts=[cells.count(i) for i in range(4)];n=len(cells)
    values=_utilities(counts,R,D,c);loo=[]
    for cell in cells:
        new=counts.copy();new[cell]-=1
        u=_utilities(new,R,D,c)
        loo.append((u['M:10'],all(u['M:10']>v for p,v in u.items() if p!='M:10'),u['blind']<=0))
    def interval(k,den):
        return dict(k=k,n=den,ci95=cp_interval(k,den) if den else None,
                    upper95=cp_interval(k,den,one_sided=True)[1] if den else None)
    individual=[(R if cell==3 else -D if cell==2 else F(0))-c for cell in cells]
    ordered=sorted(individual)
    median=(ordered[n//2-1]+ordered[n//2])/2 if n%2==0 else ordered[n//2]
    return dict(source_count=n,counts=counts,mean_credit=values,
                target_credit_median=median,target_credit_range=(min(individual),max(individual)),
                leave_one=dict(target_min=min(x[0] for x in loo),target_max=max(x[0] for x in loo),
                               target_positive=sum(x[0]>0 for x in loo),target_unique_best=sum(x[1] for x in loo),
                               best_and_blind_nonpositive=sum(x[1] and x[2] for x in loo),replicates=n),
                withheld=interval(counts[0]+counts[1],n),
                acceptance_loss=interval(counts[1],n),
                lost_given_acceptance=interval(counts[1],counts[1]+counts[3]),
                retained_given_acceptance=interval(counts[3],counts[1]+counts[3]),
                rejected_given_withheld=interval(counts[0],counts[0]+counts[1]),
                target_vs_idle_tv_zero_boundary=(values['M:10']/(R+D) if R+D>0 and values['M:10']>0 and F(counts[3],n)>=values['M:10']/(R+D) else None),
                bootstrap=exact_bootstrap(counts,R,D,c))


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    source=root/'artifacts/incentive_transfer_v1/summary.json'
    raw=source.read_bytes();summary=json.loads(raw)
    if not summary['complete'] or summary['pending'] or summary['analysis']['source_count']!=24:
        raise ValueError('Expected the complete sealed 24-source pilot')
    result=analyze(summary['analysis']['items'],F(22),F(7757,1050),F(1))
    for name,value in result['mean_credit'].items():
        if value!=F(summary['analysis']['policies'][name]['mean_credit']):
            raise AssertionError('Independent policy reconstruction differs')
    result.update(version='incentive-uncertainty-r80',source_sha256=hashlib.sha256(raw).hexdigest(),
                  contract=dict(R='22',D='7757/1050',c='1'),new_api_calls=0,
                  scope='Post-hoc fixed-contract sensitivity; iid superpopulation intervals are illustrative, not design-based finite-population inference or predeclared noninferiority. No controller independence assumed.')
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
