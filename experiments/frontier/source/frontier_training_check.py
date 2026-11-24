"""Independent probability/class checks for the fixed two-condition head pilot.

These checks were specified after observing the pilot's logit responses. They
are sensitivity analyses, not new primary endpoints or evidence of causality.
"""
from .common import write_json
from .frontier_protocol import DEST
from .frontier_criterion import hit_scores, row_correlation, interval
from .funnybirds import FB, OFFSETS
import numpy as np
import torch


def probabilities(z):
    output=[]
    for lo,hi in zip(OFFSETS[:-1],OFFSETS[1:]):
        a=z[...,lo:hi];a=np.exp(a-a.max(-1,keepdims=True))
        output.append(a/a.sum(-1,keepdims=True))
    return np.concatenate(output,-1)


def main():
    d=dict(np.load(DEST/'training_features.npz'))
    c=dict(np.load(DEST/'criterion_inputs.npz'))
    weights=c['weights'].astype(float)
    # Evaluate the original frozen species readout without fitting a new one.
    state=torch.load(FB/'readouts.pt',map_location='cpu',weights_only=False)['cbm']['state_dict']
    cw,cb=state['weight'].numpy(),state['bias'].numpy()
    meta=np.load(FB/'metadata.npz')
    y=meta['test_classes'][d['ids']];vy=meta['train_classes'][d['val_rows']]
    from .common import OUT, ROOT
    pop=np.load(OUT/'controlled_audit_2026-11-19/stage2_population.npz')
    q,a=pop['queries'],pop['alternatives']
    context=dict(np.load(ROOT/'github_cun/main/data/core_audit/data/FB_mask_context.npz'))
    original_logits=np.load(FB/'predictions.npz')['test_p'][d['ids']]@cw.T+cb
    foil=original_logits.copy();foil[np.arange(200),y]=-np.inf;foil=foil.argmax(1)
    head_indices=np.random.default_rng(2026112424).integers(3,size=(len(weights),3))
    per_condition={};arrays={};inference={}
    excess_draws={}
    for label in ['0.0','5.0']:
        result=np.load(DEST/f'training_lambda_{label}.npz')
        w,b=result['weight'],result['bias']
        z=d['test_features'].mean((-2,-1))@w.T+b
        zd=d['deleted_features'].mean((-2,-1))@w.T+b
        p,pd=probabilities(z),probabilities(zd)
        vp=probabilities(d['validation_features'].mean((-2,-1))@w.T+b)
        inference[label+'__p']=p;inference[label+'__deleted_p']=pd
        inference[label+'__validation_p']=vp
        response=np.take_along_axis(p,q,1)-np.diagonal(np.take_along_axis(pd,q[:,None],2),axis1=1,axis2=2)
        normalized=result['response']/np.linalg.norm(w[q]-w[a],axis=-1)
        logits=p@cw.T+cb
        class_response=np.einsum('bc,bkc->bk',cw[y]-cw[foil],p[:,None]-pd)
        s,g=hit_scores(context,result['peaks'])
        gain_by_image=g.reshape(4,200,5).mean(-1).T
        all_heads_draw=weights@gain_by_image/200
        excess_draws[label]=all_heads_draw[:,0]-all_heads_draw[np.arange(len(weights))[:,None],1+head_indices].mean(1)
        values={'S':s[0].reshape(200,5),'G':g[0].reshape(200,5),
                'deltaG':(g[0]-g[1:].mean(0)).reshape(200,5),
                'concept_probability_drop':response,'norm_adjusted_logit_drop':normalized,
                'class_margin_drop':class_response}
        arrays[label]=values
        per_condition[label]={'validation_species_accuracy_frozen_readout':float(((vp@cw.T+cb).argmax(1)==vy).mean()),
            'audit_species_accuracy_frozen_readout':float((logits.argmax(1)==y).mean()),
            'mean_concept_probability_drop':float(response.mean()),
            'mean_norm_adjusted_logit_drop':float(normalized.mean()),
            'mean_class_margin_drop_fixed_readout':float(class_response.mean()),
            'deltaG_probability_response_macro_rho':float(np.nanmean(row_correlation(values['deltaG'],response,axis=0)))}
    differences={}
    for name in arrays['0.0']:
        difference=arrays['5.0'][name]-arrays['0.0'][name]
        differences[name]={'paired_change':float(difference.mean()),
                           'image_only_ci95':interval(weights@difference.mean(1)/200)}
    differences['deltaG']['paired_image_head_ci95']=interval(excess_draws['5.0']-excess_draws['0.0'])
    result={'scope':'Post-outcome sensitivity: probability, row-norm and fixed species-readout responses. One frozen encoder and one training seed; held-out random heads. Not semantic interventions, a causal certificate, or a tested new method.',
        'conditions':per_condition,'regularized_minus_ce':differences,
        'decision':'Hard excess improves, but do not infer improved causal faithfulness from raw logits or score optimization. Extra annotation supervision and differentiable-surrogate controls remain necessary for a method paper.'}
    np.savez_compressed(DEST/'training_check_inputs.npz',weights=weights,head_indices=head_indices,
        **{label+'__'+name:value for label,values in arrays.items() for name,value in values.items()},
        excess_draws_0=excess_draws['0.0'],excess_draws_5=excess_draws['5.0'])
    np.savez_compressed(DEST/'training_inference.npz',**inference,queries=q,alternatives=a,head_indices=head_indices,
        validation_labels=d['val_labels'],species_labels=y,original_rivals=foil,
        readout_weight=cw,readout_bias=cb,train_rows=d['train_rows'],val_rows=d['val_rows'],ids=d['ids'])
    write_json(DEST/'training_check_results.json',result)
    print(result)


if __name__=='__main__':main()
