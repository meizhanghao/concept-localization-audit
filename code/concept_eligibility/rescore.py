"""CPU/NumPy-only reproduction from derived query scores and paired draws."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
d=np.load(root/'data.npz');a=d['attribute'];ii=d['image'];w=d['weights']
den=w@d['observed'];heads=d['heads'];ix=np.arange(len(w))[:,None]
perm=np.random.default_rng(2026112222).integers(0,32,(len(w),32))
def point(v):return float(np.mean([v[a==j].mean() for j in range(40)]))
def boot(v):
    m=np.zeros((800,40));m[ii,a]=v
    return ((w@m)/den).mean(1)
g0=d['correct_own']-d['correct_donor'];g1=d['wrong_own']-d['wrong_donor']
value=g0-g1.mean(1);draws=[]
for h in range(6):
    bank=np.stack([boot(v) for v in g1[h]],1)
    draws.append(boot(g0[h])-bank[ix,perm].mean(1))
draws=np.array(draws).T
adjusted=draws[:,0]-draws[:,1:][ix,heads].mean(1)
primary={'G_specific':point(value[0]),'ci95':np.quantile(draws[:,0],[.025,.975]).tolist(),
         'adjusted':point(value[0]-value[1:].mean(0)),
         'adjusted_ci95':np.quantile(adjusted,[.025,.975]).tolist()}
pid=d['pair_id'];im=d['pair_image'];pw=d['pair_weights'];n=len(pw)
dm=np.zeros((800,n));np.add.at(dm,(im,pid),1);bd=w@dm;ok=(bd>0).all(1)
def cp_point(v):return float(sum(pw[j]*v[pid==j].mean() for j in range(n)))
def cp_boot(v):
    m=np.zeros_like(dm);np.add.at(m,(im,pid),v)
    return (((w[ok]@m)/bd[ok])*pw).sum(1)
cg=d['pair_own']-d['pair_donor'];ce=cg[:,0]-cg[:,1]
cb=np.array([cp_boot(v) for v in ce]).T;hi=heads[ok];ci=np.arange(ok.sum())[:,None]
cadj=cb[:,0]-cb[:,1:][ci,hi].mean(1)
copresent={'G_specific':cp_point(ce[0]),'ci95':np.quantile(cb[:,0],[.025,.975]).tolist(),
           'adjusted':cp_point(ce[0]-ce[1:].mean(0)),
           'adjusted_ci95':np.quantile(cadj,[.025,.975]).tolist()}
out={'primary':primary,'both_present':copresent}
expected=json.loads((root/'expected.json').read_text())
errors=[]
for cohort in out:
    for k in out[cohort]:
        err=float(np.max(np.abs(np.asarray(out[cohort][k])-expected[cohort][k])))
        errors.append(err)
assert max(errors)<1e-12,errors
out.update(status='passed',maximum_error=max(errors))
(root/'results.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
