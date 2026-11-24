"""Distinguish renderer response interactions from latent concept interactions."""
from .common import OUT, configure, write_json
from .frontier_protocol import DEST
from .funnybirds import FB, ZIP, PARTS, OFFSETS, image, probability
from .models import ImageConceptModel
import itertools
import json
import zipfile
import numpy as np
import torch


def main():
    configure(0)
    model = ImageConceptModel(count=31).cuda().eval()
    model.load_state_dict(torch.load(FB/'model.pt',map_location='cpu',weights_only=False)['state_dict'])
    pop = dict(np.load(OUT/'controlled_audit_2026-11-19/stage2_population.npz'))
    ids = pop['ids']
    pairs = list(itertools.combinations(range(5),2))
    semantic = np.load(FB/'semantic_predictions.npz')['deletion_p'][ids]
    predictions = np.load(FB/'predictions.npz')['test_p'][ids]
    labels = np.load(FB/'metadata.npz')['test_classes'][ids]
    readout = torch.load(FB/'readouts.pt',map_location='cpu',weights_only=False)['cbm']['state_dict']
    W,b = readout['weight'].numpy(),readout['bias'].numpy()
    logits = predictions@W.T+b
    rival_logits = logits.copy();rival_logits[np.arange(200),labels]=-np.inf
    rivals = rival_logits.argmax(1)
    contrast = W[labels]-W[rivals]
    single = np.einsum('bc,bkc->bk',contrast,predictions[:,None]-semantic)
    cache = DEST/'pair_deleted_probabilities.npz'
    if cache.exists():
        double = np.load(cache)['p']
    else:
        double = []
        with zipfile.ZipFile(ZIP) as z, torch.inference_mode():
            parameters = json.loads(z.read('FunnyBirds/dataset_test.json'))
            for a,bpart in pairs:
                images=[]
                kept = sorted(set(PARTS)-{PARTS[a],PARTS[bpart]})
                for idx in ids:
                    r = parameters[idx]
                    name = f'FunnyBirds/test_interventions/{r["class_idx"]}/{idx:06d}/body_'+ '_'.join(kept)+'.png'
                    images.append(image(z,name))
                values=[]
                for start in range(0,200,64):
                    x=torch.as_tensor(np.stack(images[start:start+64]),device='cuda').float().div_(255)
                    values.append(probability(model(x)).cpu().numpy())
                double.append(np.concatenate(values))
                print('renderer pair',PARTS[a],PARTS[bpart],flush=True)
        double=np.stack(double,axis=1)
        np.savez_compressed(cache,p=double,ids=ids,pairs=np.array(pairs))
    second=[];latent=[]
    for j,(a,bpart) in enumerate(pairs):
        combined = np.einsum('bc,bc->b',contrast,predictions-double[:,j])
        second.append(single[:,a]+single[:,bpart]-combined)
        # Replace only each designated concept group in the additive readout.
        da=np.zeros_like(predictions,dtype=np.float64);db=np.zeros_like(da)
        lo,hi=OFFSETS[a:a+2];da[:,lo:hi]=semantic[:,a,lo:hi]-predictions[:,lo:hi]
        lo,hi=OFFSETS[bpart:bpart+2];db[:,lo:hi]=semantic[:,bpart,lo:hi]-predictions[:,lo:hi]
        p=predictions.astype(np.float64);w=contrast.astype(np.float64)
        f0=np.einsum('bc,bc->b',w,p)
        fa=np.einsum('bc,bc->b',w,p+da);fb=np.einsum('bc,bc->b',w,p+db)
        fab=np.einsum('bc,bc->b',w,p+da+db)
        latent.append(f0-fa-fb+fab)
    second=np.stack(second,1);latent=np.stack(latent,1)
    weights=np.load(DEST/'criterion_inputs.npz')['weights'].astype(float)
    mean_abs=np.abs(second).mean(1)
    draws=weights@mean_abs/200
    results={'scope':'All ten part pairs, same 200 R18 images; official same-scene both-deleted renders. Signed fixed true-class-versus-original-rival logit margin.',
             'renderer_mean_absolute_pair_interaction':float(mean_abs.mean()),
             'image_cluster_ci95':np.quantile(draws,[.025,.975]).tolist(),
             'per_pair_mean_absolute':{f'{PARTS[a]}+{PARTS[b]}':float(np.abs(second[:,j]).mean()) for j,(a,b) in enumerate(pairs)},
             'latent_additive_logit_max_abs_interaction':float(np.abs(latent).max()),
             'single_mean_abs_response':float(np.abs(single).mean()),
             'decision':'Raw-image interactions exist but are not latent-concept interactions. A union hit does not identify either; no interaction-localization metric added.',
             'not_claimed':['A novel Shapley interaction method','Real-world semantic causal effects','AND/OR concept learning inferred from union masks']}
    assert np.abs(latent).max()<1e-10
    np.savez_compressed(DEST/'interaction_inputs.npz',renderer=second,latent=latent,singles=single,pairs=np.array(pairs),ids=ids)
    write_json(DEST/'interaction_results.json',results)
    print(results)


if __name__=='__main__':
    main()
