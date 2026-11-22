"""CPU/NumPy-only rescoring of frozen concept-localization audit records.

No project imports, network, raw images, model weights, PyTorch or GPU required.
Run: python rescore.py --output results.json
"""
from pathlib import Path
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import argparse,hashlib,json,time,platform
import numpy as np


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def hit_xy(peaks,points):
    lo=np.maximum(peaks[...,None,:]-45,0);hi=np.minimum(peaks[...,None,:]+45,224)
    return ((points>=lo)&(points<=hi)).all(-1).any(-1)


def summarize(case,root):
    a=dict(np.load(root/case['shared']));a.update(dict(np.load(root/case['file'])))
    peaks=a['peaks'];donors=a['donors'];image=a['image'];group=a['group'];weights=a['weights'];groups=int(group.max()+1)
    # Query populations in the bundle are already restricted to the declared
    # macro groups; images with zero eligible pairs retain their bootstrap rows.
    counts=np.bincount(group,minlength=groups);assert (counts>0).all()
    observed=np.zeros((weights.shape[1],groups));observed[image,group]=1
    den=weights@observed;assert (den>0).all()
    alpha=1/(groups*counts[group]);scores=[];gains=[];boots=[]
    if 'windows_packed' in a:
        side=int(a['side']);w=np.unpackbits(a['windows_packed'],axis=1,count=side*side).astype(bool)
        def hit(p):return w[np.arange(len(image)),p[:,0]*side+p[:,1]]
        def donor_hit(p):
            d=p[donors];return w[np.arange(len(image))[None,:],d[...,0]*side+d[...,1]].mean(0)
    else:
        def hit(p):return hit_xy(p,a['points_xy'])
        def donor_hit(p):return hit_xy(p[donors],a['points_xy'][None]).mean(0)
    for p in peaks:
        own=hit(p).astype(float);gain=own-donor_hit(p)
        matrix=np.zeros_like(observed);matrix[image,group]=gain
        scores.append(float(alpha@own));gains.append(gain);boots.append(((weights@matrix)/den).mean(1))
    gains=np.array(gains);boots=np.array(boots);point=gains@alpha
    result={'S':scores[0],'G':float(point[0]),'G_ci95':np.quantile(boots[0],[.025,.975]).tolist()}
    if len(peaks)>1:
        random=point[1:].mean();fixed=boots[0]-boots[1:].mean(0)
        result.update(G_random=float(random),delta_G=float(point[0]-random),
                      fixed_head_image_ci95=np.quantile(fixed,[.025,.975]).tolist())
        if case.get('joint',False):
            head=a['head_indices'];b=np.arange(len(weights))[:,None]
            joint=boots[0]-boots[1:].T[b,head].mean(1)
            result['image_head_ci95']=np.quantile(joint,[.025,.975]).tolist()
    return result,boots[0]


def compare(actual,expected,path=''):
    errors=[]
    for key,value in expected.items():
        if isinstance(value,dict):errors.extend(compare(actual[key],value,path+'/'+key))
        elif isinstance(value,(list,float,int)):
            error=float(np.max(np.abs(np.asarray(actual[key])-np.asarray(value))))
            if error>1e-12:errors.append({'field':path+'/'+key,'error':error})
    return errors


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results.json');args=parser.parse_args()
    root=Path(__file__).resolve().parent;start=time.perf_counter();manifest=json.loads((root/'manifest.json').read_text())
    for name,sha in manifest['sha256'].items():assert digest(root/name)==sha,('File changed',name)
    expected=json.loads((root/'expected_results.json').read_text());actual={};gain_draws={}
    for case in manifest['cases']:
        actual[case['name']],gain_draws[case['name']]=summarize(case,root)
        print(case['name'],json.dumps(actual[case['name']]),flush=True)
    for comparison in manifest.get('paired_comparisons',[]):
        x,y=comparison['first'],comparison['second']
        actual[comparison['name']]={'gain_difference':actual[x]['G']-actual[y]['G'],
            'paired_image_ci95':np.quantile(gain_draws[x]-gain_draws[y],[.025,.975]).tolist()}
    errors=compare(actual,expected);assert not errors,errors
    report={'status':'passed','cases':actual,'maximum_allowed_numeric_difference':1e-12,
            'runtime_seconds':time.perf_counter()-start,'python':platform.python_version(),'numpy':np.__version__,
            'scope':manifest['scope']}
    output=Path(args.output);output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','runtime_seconds','python','numpy']}))


if __name__=='__main__':main()
