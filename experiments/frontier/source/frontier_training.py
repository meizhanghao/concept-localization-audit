"""Small frozen-backbone training gate, not a new training-method benchmark."""
from .common import OUT, configure, write_json
from .frontier_protocol import DEST
from .frontier_criterion import hit_scores, row_correlation
from .funnybirds import FB, ZIP, PARTS, COUNTS, OFFSETS, COLORS, labels
from .models import ImageConceptModel
import json
import zipfile
import io
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter
import torch
from torch.nn import functional as F


def encode(model, rgb, rows):
    result=[]
    with torch.inference_mode():
        for start in range(0,len(rows),64):
            x=torch.as_tensor(np.array(rgb[rows[start:start+64]]),device='cuda').float().div_(255)
            result.append(model(x,True)[2].cpu().numpy())
    return np.concatenate(result)


def get_cache():
    path=DEST/'training_features.npz'
    if path.exists():return dict(np.load(path))
    meta=np.load(FB/'metadata.npz');rng=np.random.default_rng(2026112410)
    train=np.concatenate([rng.choice(np.flatnonzero((meta['train_classes']==c)&~meta['validation']),40,replace=False) for c in range(50)])
    val=np.concatenate([rng.choice(np.flatnonzero((meta['train_classes']==c)&meta['validation']),20,replace=False) for c in range(50)])
    assert not np.intersect1d(train,val).size
    model=ImageConceptModel(count=31).cuda().eval()
    model.load_state_dict(torch.load(FB/'model.pt',map_location='cpu',weights_only=False)['state_dict'])
    train_rgb=np.load(FB/'train_rgb.npy',mmap_mode='r')
    test_rgb=np.load(FB/'test_rgb.npy',mmap_mode='r')
    pop=np.load(OUT/'controlled_audit_2026-11-19/stage2_population.npz')
    ids=pop['ids']
    feats=encode(model,train_rgb,train);validation=encode(model,train_rgb,val)
    test=encode(model,test_rgb,ids)
    deleted=np.load(FB/'deleted_rgb.npy',mmap_mode='r')[ids].reshape(1000,3,160,160)
    deletion=encode(model,deleted,np.arange(1000)).reshape(200,5,512,5,5)
    windows=[]
    with zipfile.ZipFile(ZIP) as z:
        params=json.loads(z.read('FunnyBirds/dataset_train.json'))
        for at,index in enumerate(train):
            name=f'FunnyBirds/train_part_map/{params[index]["class_idx"]}/{index:06d}.png'
            with Image.open(io.BytesIO(z.read(name))) as im:a=np.asarray(im.convert('RGB'))
            row=[]
            for part in PARTS:
                mask=np.any([np.all(a==np.array(color),axis=-1) for color in COLORS[part]],axis=0)
                mask=np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((160,160),Image.Resampling.NEAREST))
                window=maximum_filter(mask,size=49)>0
                row.append(window.reshape(5,32,5,32).mean((1,3)))
            windows.append(row)
    data=dict(train_features=feats,validation_features=validation,test_features=test,deleted_features=deletion,
              windows=np.asarray(windows,dtype=np.float32),train_rows=train,val_rows=val,ids=ids,
              train_labels=meta['train_labels'][train],val_labels=meta['train_labels'][val],test_labels=meta['test_labels'][ids])
    np.savez_compressed(path,**data)
    print('training features prepared',flush=True)
    return data


def alternatives(y):
    return np.stack([1+y[:,j]%(COUNTS[j]-1) for j in range(5)],axis=1)+OFFSETS[:-1]


def peaks(features,weight,q,a):
    f=torch.as_tensor(features,device='cuda');w=torch.as_tensor(weight,device='cuda')
    maps=torch.einsum('bqc,bchw->bqhw',w[q]-w[a],f).relu()
    maps=F.interpolate(maps,(160,160),mode='bilinear',align_corners=False)
    p=maps.flatten(2).argmax(-1).cpu().numpy()
    return np.stack([p//160,p%160],-1).reshape(1000,2).astype(np.int16)


def main():
    configure(0)
    data=get_cache()
    saved=torch.load(FB/'model.pt',map_location='cpu',weights_only=False)['state_dict']
    initial_w=saved['head.weight'].clone().cuda();initial_b=saved['head.bias'].clone().cuda()
    x=torch.as_tensor(data['train_features'],device='cuda')
    win=torch.as_tensor(data['windows'],device='cuda')
    y=torch.as_tensor(data['train_labels'],device='cuda')
    q=y+torch.as_tensor(OFFSETS[:-1],device='cuda')
    alt=torch.as_tensor(alternatives(data['train_labels']),device='cuda')
    gen=torch.Generator(device='cuda').manual_seed(821)
    refs=[]
    for _ in range(3):
        r=torch.randn(initial_w.shape,device='cuda',generator=gen)
        refs.append(r/r.norm(dim=1,keepdim=True)*initial_w.norm(dim=1,keepdim=True))
    refs=torch.stack(refs)
    # Numerical gradient gate: both active hinges differ only by a constant.
    a=torch.tensor(.01,requires_grad=True);r=torch.tensor(.03)
    grad_d=torch.autograd.grad(F.relu(.1-a),a,retain_graph=True)[0]
    grad_r=torch.autograd.grad(F.relu(.1-(a-r)),a)[0]
    assert float(abs(grad_d-grad_r))==0
    result={'scope':'Head-only ten-epoch feasibility pilot, 2000 training/1000 validation/200 audit images. Extra localization supervision; no test-selected epoch or lambda.',
            'fixed_reference_active_hinge_gradient_difference':float(abs(grad_d-grad_r)), 'conditions':{}}
    for lam in [0.,5.]:
        configure(2026112411)
        W=torch.nn.Parameter(initial_w.clone());b=torch.nn.Parameter(initial_b.clone())
        optimizer=torch.optim.Adam([W,b],lr=.001)
        history=[]
        for ep in range(10):
            order=np.random.default_rng(2026112412+ep).permutation(len(x));losses=[];soft=[]
            for start in range(0,len(order),64):
                idx=torch.as_tensor(order[start:start+64],device='cuda')
                feat=x[idx];target=y[idx];query=q[idx];foil=alt[idx];window=win[idx]
                logits=F.linear(feat.mean((-2,-1)),W,b)
                ce=sum(F.cross_entropy(logits[:,lo:hi],target[:,j]) for j,(lo,hi) in enumerate(zip(OFFSETS[:-1],OFFSETS[1:])))/5
                donor=torch.arange(len(idx),device='cuda')[:,None].expand(-1,5).clone()
                valid=torch.zeros((len(idx),5),device='cuda',dtype=torch.bool)
                for j in range(5):
                    for label in query[:,j].unique():
                        rows=torch.where(query[:,j]==label)[0]
                        if len(rows)>1:
                            donor[rows,j]=rows.roll(1);valid[rows,j]=True
                valid &= window.sum((-2,-1))>0
                ww=W[query]-W[foil]
                maps=torch.einsum('bqc,bchw->bqhw',ww,feat)
                other=torch.einsum('bqc,bqchw->bqhw',ww,feat[donor])
                own_mass=(maps.flatten(2).softmax(-1)*window.flatten(2)).sum(-1)
                donor_mass=(other.flatten(2).softmax(-1)*window.flatten(2)).sum(-1)
                g=own_mass-donor_mass
                with torch.no_grad():
                    rw=refs[:,query]-refs[:,foil]
                    rm=torch.einsum('nbqc,bchw->nbqhw',rw,feat)
                    rd=torch.einsum('nbqc,bqchw->nbqhw',rw,feat[donor])
                    reference=((rm.flatten(3).softmax(-1)-rd.flatten(3).softmax(-1))*window.flatten(2)[None]).sum(-1).mean(0)
                spatial=F.relu(.1-(g-reference))[valid].mean()
                loss=ce+lam*spatial
                optimizer.zero_grad();loss.backward();optimizer.step()
                losses.append(float(ce.detach()));soft.append(float((g-reference)[valid].mean().detach()))
            history.append({'epoch':ep+1,'concept_ce':float(np.mean(losses)),'soft_excess':float(np.mean(soft))})
        wn,bn=W.detach().cpu().numpy(),b.detach().cpu().numpy()
        test_z=data['test_features'].mean((-2,-1))@wn.T+bn
        val_z=data['validation_features'].mean((-2,-1))@wn.T+bn
        pred=np.stack([test_z[:,lo:hi].argmax(1) for lo,hi in zip(OFFSETS[:-1],OFFSETS[1:])],1)
        vp=np.stack([val_z[:,lo:hi].argmax(1) for lo,hi in zip(OFFSETS[:-1],OFFSETS[1:])],1)
        pop=np.load(OUT/'controlled_audit_2026-11-19/stage2_population.npz')
        qt,at=pop['queries'],pop['alternatives']
        pp=[peaks(data['test_features'],wn,qt,at)]
        for seed in [2026112421,2026112422,2026112423]:
            rng=np.random.default_rng(seed);rw=rng.standard_normal(wn.shape).astype(np.float32)
            rw=rw/np.linalg.norm(rw,axis=1,keepdims=True)*np.linalg.norm(wn,axis=1,keepdims=True)
            pp.append(peaks(data['test_features'],rw,qt,at))
        from .common import ROOT
        context=dict(np.load(ROOT/'github_cun/main/data/core_audit/data/FB_mask_context.npz'))
        own,gain=hit_scores(context,np.stack(pp))
        zz=data['deleted_features'].mean((-2,-1))@wn.T+bn
        own_margin=np.take_along_axis(test_z,qt,1)-np.take_along_axis(test_z,at,1)
        after=np.take_along_axis(zz,qt[:,None],2)-np.take_along_axis(zz,at[:,None],2)
        response=own_margin-np.diagonal(after,axis1=1,axis2=2)
        delta=(gain[0]-gain[1:].mean(0)).reshape(200,5)
        result['conditions'][str(lam)]={'validation_concept_accuracy':float((vp==data['val_labels']).mean()),
            'audit_concept_accuracy':float((pred==data['test_labels']).mean()),
            'S':float(own[0].mean()),'G':float(gain[0].mean()),'deltaG_heldout_reference':float(delta.mean()),
            'mean_signed_concept_deletion_response':float(response.mean()),
            'deltaG_response_macro_rho':float(np.nanmean(row_correlation(delta,response,axis=0))), 'history':history}
        np.savez_compressed(DEST/f'training_lambda_{lam}.npz',weight=wn,bias=bn,peaks=np.stack(pp),response=response)
        print('training condition',lam,result['conditions'][str(lam)],flush=True)
    result['decision']='Feasibility/gaming gate only. Loss uses annotation-assisted soft native-grid mass, not hard peak scoring. No novel training-method or causal-improvement claim.'
    write_json(DEST/'training_results.json',result)


if __name__=='__main__':main()
