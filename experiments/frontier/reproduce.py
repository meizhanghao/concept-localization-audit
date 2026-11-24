"""Recompute the finite-reference and head-pilot results from frozen arrays.

Run with Python 3.10+ and NumPy. No expected-result JSON is read. These are
conditional numerical reproductions, not raw training or a causal evaluation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

BASE=Path(__file__).resolve().parent


def verify_manifest():
    """Verify the frozen inputs and executed sources before loading arrays."""
    manifest = json.loads((BASE / 'manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['files'].items():
        path = (BASE / relative).resolve()
        if not path.is_relative_to(BASE.resolve()) or not path.is_file():
            raise ValueError(f'Invalid or missing manifest entry: {relative}')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'SHA-256 mismatch: {relative}')


def ci(draws):
    return np.quantile(draws,[.025,.975]).tolist()


def gains(context,peaks):
    """Every transferred source peak is scored in the RECIPIENT window."""
    side=int(context['side'])
    windows=np.unpackbits(context['windows_packed'],axis=1,count=side*side).astype(bool)
    if peaks.ndim==2:peaks=peaks[None]
    scores=[];own=[]
    for p in peaks:
        pos=p[:,0]*side+p[:,1]
        correct=windows[np.arange(len(p)),pos].astype(float)
        donor_hit=windows[np.arange(len(p))[None],pos[context['donors']]]
        own.append(correct)
        scores.append(correct-donor_hit.mean(0))
        # Check source/target indexing independently on a fixed subset.
        for q in [0,17,205,633,999]:
            direct=np.mean([windows[q,int(pos[source])] for source in context['donors'][:,q]])
            assert abs(direct-float(donor_hit[:,q].mean()))<1e-14
    return np.asarray(own),np.asarray(scores)


def reference_order(context,data):
    weights=context['weights'].astype(float)
    n=200;b=len(weights)
    trained={}
    for m,name in enumerate(['cam','gradient']):
        _,g=gains(context,data['trained'][m])
        trained[name]=g.reshape(n,5).mean(1)
    points=[];draws=[];rows={}
    for j,label in enumerate(data['bank_names'].tolist()):
        values={}
        heads=data['head_indices'][j]
        for m,name in enumerate(['cam','gradient']):
            _,g=gains(context,data['random'][j,m])
            per_image=g.reshape(3,n,5).mean(-1).T
            reference=weights@per_image/n
            boot=weights@trained[name]/n-reference[np.arange(b)[:,None],heads].mean(1)
            values[name]=(float(trained[name].mean()-per_image.mean()),boot)
        point=values['cam'][0]-values['gradient'][0]
        boot=values['cam'][1]-values['gradient'][1]
        rows[label]={'cam':values['cam'][0],'gradient':values['gradient'][0],
                     'cam_minus_gradient':point,'paired_ci95':ci(boot)}
        points.append(point);draws.append(boot)
    points=np.array(points);draws=np.stack(draws,1)
    sd=draws.std(0,ddof=1)
    q=np.quantile(np.max(np.abs((draws-points)/sd),axis=1),.95)
    bounds=np.stack([points-q*sd,points+q*sd],1)
    for name,bound in zip(rows,bounds):rows[name]['simultaneous_band']=bound.tolist()
    return {'banks':rows,'point_envelope':[float(points.min()),float(points.max())],
        'simultaneous_envelope':[float(bounds[:,0].min()),float(bounds[:,1].max())],
        'ranking_resolved':bool((bounds[:,0]>0).all() or (bounds[:,1]<0).all())}


def training(context,raw):
    weights=context['weights'].astype(float);n=200;b=len(weights)
    heads=raw['head_indices']
    y=raw['species_labels'];foils=raw['original_rivals']
    w=raw['readout_weight'].astype(float)
    q=raw['queries'];rows={};values={};boots={}
    offsets=np.array([0,5,9,14,24,31])
    for label in ['0.0','5.0']:
        cached=np.load(BASE/'data'/f'training_lambda_{label}.npz',allow_pickle=False)
        s,g=gains(context,cached['peaks'])
        by_image=g.reshape(4,n,5).mean(-1).T
        sampled=weights@by_image/n
        boots[label]=sampled[:,0]-sampled[np.arange(b)[:,None],1+heads].mean(1)
        p=raw[label+'__p'].astype(float);pd=raw[label+'__deleted_p'].astype(float)
        response=np.take_along_axis(p,q,1)-np.diagonal(np.take_along_axis(pd,q[:,None],2),axis1=1,axis2=2)
        class_response=np.einsum('bc,bkc->bk',w[y]-w[foils],p[:,None]-pd)
        vp=raw[label+'__validation_p']
        pred=np.stack([vp[:,lo:hi].argmax(1) for lo,hi in zip(offsets[:-1],offsets[1:])],1)
        delta=(g[0]-g[1:].mean(0)).reshape(n,5)
        values[label]={'deltaG':delta,'probability_drop':response,'class_margin_drop':class_response}
        rows[label]={'S':float(s[0].mean()),'G':float(g[0].mean()),'deltaG':float(delta.mean()),
            'validation_concept_accuracy':float((pred==raw['validation_labels']).mean()),
            'probability_drop':float(response.mean()),'class_margin_drop':float(class_response.mean())}
    changes={}
    for name in values['0.0']:
        d=values['5.0'][name]-values['0.0'][name]
        changes[name]={'paired_change':float(d.mean()),'image_only_ci95':ci(weights@d.mean(1)/n)}
    changes['deltaG']['paired_image_head_ci95']=ci(boots['5.0']-boots['0.0'])
    return {'conditions':rows,'regularized_minus_ce':changes,
        'scope':'One fixed encoder/training seed; three held-out heads. Response endpoints selected after primary training outcomes. No equivalence, semantic causality or new learning-method claim.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    verify_manifest()
    context=dict(np.load(BASE/'data/context.npz',allow_pickle=False))
    assert np.array_equal(context['image'],np.repeat(np.arange(200),5))
    assert np.array_equal(context['group'],np.tile(np.arange(5),200))
    assert np.all(context['weights'].sum(1)==200)
    reference=dict(np.load(BASE/'data/reference_inputs.npz',allow_pickle=False))
    raw=dict(np.load(BASE/'data/training_inference.npz',allow_pickle=False))
    result={'reference_order':reference_order(context,reference),'training':training(context,raw)}
    text=json.dumps(result,indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(text+'\n',encoding='utf-8')


if __name__=='__main__':main()
