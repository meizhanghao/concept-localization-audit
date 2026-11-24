"""FunnyBirds semantic and response pilot. No new dataset or new-method claim.

Official labels and split are read from the supplied zip. Validation is carved
from official training data; official test is never used for model selection.
"""
from .common import BASE, OUT, ROOT, configure, write_json
from .models import ImageConceptModel
import argparse
import ast
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageFilter
from scipy.stats import rankdata

FB = OUT/'funnybirds_2026-11-15'
ZIP = ROOT/'main_python/dataset/fbdl/FunnyBirds.zip'
SIDE = 160
PARTS = ['beak','eye','foot','tail','wing']
COUNTS = [5,4,5,10,7]  # Each includes a separate absence category.
OFFSETS = np.r_[0,np.cumsum(COUNTS)]
COLORS = {'beak':[(255,255,0)],'eye':[(255,255,253),(255,255,254)],'foot':[(255,0,1),(255,0,2)],'tail':[(0,0,255)],'wing':[(0,255,1),(0,255,2)]}
CYCLE = {'red':'green','green':'blue','blue':'red'}


def save(name,data):
    write_json(FB/name,data)


def interval(v):
    v=np.asarray(v,float);v=v[np.isfinite(v)]
    if not len(v):return {'mean':None,'ci95':[None,None],'images':0}
    rng=np.random.default_rng(842)
    means=np.array([v[rng.integers(len(v),size=len(v))].mean() for _ in range(2000)])
    return {'mean':float(v.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'images':len(v)}


def correlations(a,b):
    ar=rankdata(a,axis=-1);br=rankdata(b,axis=-1)
    ar-=ar.mean(-1,keepdims=True);br-=br.mean(-1,keepdims=True)
    denominator=np.sqrt((ar*ar).sum(-1)*(br*br).sum(-1))
    out=np.divide((ar*br).sum(-1),denominator,out=np.full_like(denominator,np.nan),where=denominator>1e-10)
    n=np.isfinite(out).sum(-1)
    return np.divide(np.nansum(out,-1),n,out=np.full(len(n),np.nan),where=n>0)


def image(z,name):
    with Image.open(io.BytesIO(z.read(name))) as im:
        return np.asarray(im.convert('RGB').resize((SIDE,SIDE),Image.Resampling.BILINEAR)).transpose(2,0,1).copy()


def labels(params,parts):
    values=[]
    for record in params:
        row=[]
        for p in PARTS:
            specification={key:record[p+'_'+key] for key in parts[p][0]}
            row.append(0 if specification['model']=='placeholder' else parts[p].index(specification)+1)
        values.append(row)
    return np.array(values,np.int64)


def prepare():
    FB.mkdir(parents=True,exist_ok=True)
    if (FB/'prepared.json').exists():return
    started=time.time()
    with zipfile.ZipFile(ZIP) as z:
        tr=json.loads(z.read('FunnyBirds/dataset_train.json'));te=json.loads(z.read('FunnyBirds/dataset_test.json'))
        parts=json.loads(z.read('FunnyBirds/parts.json'));classes=json.loads(z.read('FunnyBirds/classes.json'))
        yy=labels(tr,parts);yt=labels(te,parts)
        cls=np.array([p['class_idx'] for p in tr]);ct=np.array([p['class_idx'] for p in te])
        val=np.zeros(len(tr),bool);rng=np.random.default_rng(482)
        for c in range(50):val[rng.choice(np.flatnonzero(cls==c),100,replace=False)]=True
        assert len(tr)==50000 and len(te)==500
        # Metadata, not source-class templates, supervises partially deleted train images.
        np.savez_compressed(FB/'metadata.npz',train_labels=yy,test_labels=yt,train_classes=cls,test_classes=ct,validation=val)
        save('parts.json',parts);save('classes.json',classes)
        for split,params in [('train',tr),('test',te)]:
            cache=np.lib.format.open_memmap(FB/(split+'_rgb.npy'),mode='w+',dtype=np.uint8,shape=(len(params),3,SIDE,SIDE))
            for i,r in enumerate(params):
                cache[i]=image(z,f'FunnyBirds/{split}/{r["class_idx"]}/{i:06d}.png')
                if i%10000==0:print(json.dumps({'prepare':split,'images':i,'seconds':round(time.time()-started,1)}),flush=True)
            cache.flush();del cache
        masks=np.zeros((len(te),5,SIDE,SIDE),np.uint8)
        removed=np.lib.format.open_memmap(FB/'deleted_rgb.npy',mode='w+',dtype=np.uint8,shape=(len(te),5,3,SIDE,SIDE))
        full_render_errors=[]
        names=set(z.namelist());missing=[]
        for i,r in enumerate(te):
            prefix=f'FunnyBirds/test_interventions/{r["class_idx"]}/{i:06d}/'
            with Image.open(io.BytesIO(z.read(f'FunnyBirds/test_part_map/{r["class_idx"]}/{i:06d}.png'))) as im:
                a=np.asarray(im.convert('RGB'))
            for j,p in enumerate(PARTS):
                mask=np.any(np.stack([np.all(a==np.array(color),axis=-1) for color in COLORS[p]]),axis=0)
                masks[i,j]=np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((SIDE,SIDE),Image.Resampling.NEAREST))
                name=prefix+'body_'+'_'.join(sorted(set(PARTS)-{p}))+'.png'
                if name not in names:missing.append(name)
                else:removed[i,j]=image(z,name)
            full=prefix+'body_'+'_'.join(sorted(PARTS))+'.png'
            if full in names:
                original=np.load(FB/'test_rgb.npy',mmap_mode='r')[i]
                full_render_errors.append(float(np.abs(original.astype(float)-image(z,full)).max()))
        assert not missing,missing[:5]
        removed.flush();del removed
        np.save(FB/'test_masks.npy',masks)
    save('prepared.json',{'zip':str(ZIP),'zip_bytes':ZIP.stat().st_size,'train':len(tr),'validation':int(val.sum()),'test':len(te),'image_side':SIDE,'train_absent_fraction_per_group':(yy==0).mean(0).tolist(),'all_required_real_deletion_images_found':True,'maximum_difference_full_kept_render_vs_original_uint8':max(full_render_errors),'part_map_decoding':'Exact native RGB colors from official dataset loader, merged bilateral components, then nearest resize. Native antialiased boundary pixels not assigned by exact color equality.','seconds':time.time()-started})
    save('protocol.json',{'seed':0,'resolution':SIDE,'train':45000,'validation':5000,'test':500,'audit_indices':'First four official test images per class, 200 images; fixed before any test results.','concepts':'Five categorical groups, 26 present variants and five explicit absent categories, 31 outputs. Tail/wing variants include geometry plus color.','encoder_training':'ImageNet ResNet18, independent concept-only cross entropy, no class or localization loss, no mask as model input. Checkpoint selected on validation concept CE.','readout':'Frozen soft concepts -> learned linear 50-class head; train only fully present training images because source class of partial birds can be ambiguous. Matched frozen feature linear head sees same rows.','explanations':'True vs alternative variant logit contrast within each part. For tail/wing, alternative has same geometry and next color. All queries reported, no correct-only filtering.','evaluation':['Original concept and class prediction quality','Official pre-rendered single-part deletions: missing-category detection and preservation of the other four groups','Auxiliary masked cyclic RGB recolor on tail/wing: expected color switch, geometry and other-group preservation; not a re-rendered counterfactual','Spatial localization of contrast gradients/Grad-CAM/random-head controls','Five-region contrast response ranks for real deletion, with explicitly separated generic map and paired-image directional predictors','Exact linear class readout decomposition; algebraic closure does not prove causal semantics'],'caveats':'Pilot, one seed, reduced resolution; no original-framework headline score reproduction or novelty claim. Color edit is a pixel probe with imperfect boundary masks, not renderer ground truth.','sources':['https://github.com/visinf/funnybirds-framework','https://raw.githubusercontent.com/visinf/funnybirds-framework/main/datasets/funny_birds.py','https://raw.githubusercontent.com/visinf/funnybirds-framework/main/evaluation_protocols.py']})


def probability(logits):
    return torch.cat([logits[:,a:b].softmax(-1) for a,b in zip(OFFSETS[:-1],OFFSETS[1:])],1)


def tensor(cache,indices):
    return torch.from_numpy(np.asarray(cache[indices]).copy()).cuda().float().div_(255)


def evaluate(model,cache,yy,rows):
    total=0.;preds=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for start in range(0,len(rows),192):
            idx=rows[start:start+192];logits=model(tensor(cache,idx));y=torch.tensor(yy[idx],device='cuda')
            loss=sum(F.cross_entropy(logits[:,a:b],y[:,j],reduction='sum') for j,(a,b) in enumerate(zip(OFFSETS[:-1],OFFSETS[1:])))
            total+=loss.item();preds.append(np.stack([logits[:,a:b].argmax(-1).cpu().numpy() for a,b in zip(OFFSETS[:-1],OFFSETS[1:])],1))
    pred=np.concatenate(preds)
    return {'ce':total/(len(rows)*5),'group_accuracy':(pred==yy[rows]).mean(0).tolist(),'mean_accuracy':float((pred==yy[rows]).mean())}


def train(epochs):
    if (FB/'model.pt').exists():return
    configure(0);meta=np.load(FB/'metadata.npz');yy=meta['train_labels'];val=meta['validation']
    cache=np.load(FB/'train_rgb.npy',mmap_mode='r');tr=np.flatnonzero(~val);va=np.flatnonzero(val)
    model=ImageConceptModel(count=sum(COUNTS),pretrained=True).cuda()
    opt=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':.0003},{'params':model.head.parameters(),'lr':.001}],weight_decay=.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    scaler=torch.amp.GradScaler('cuda');best=float('inf');stale=0;history=[];started=time.time()
    for ep in range(epochs):
        model.train();order=np.random.permutation(tr);total=0.
        for start in range(0,len(order),128):
            idx=order[start:start+128];x=tensor(cache,idx);y=torch.tensor(yy[idx],device='cuda')
            flip=torch.rand(len(x),device='cuda')<.5;x[flip]=x[flip].flip(-1)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(x);loss=sum(F.cross_entropy(logits[:,a:b],y[:,j]) for j,(a,b) in enumerate(zip(OFFSETS[:-1],OFFSETS[1:])))/5
            scaler.scale(loss).backward();scaler.step(opt);scaler.update();total+=loss.item()*len(idx)
        model.eval();score=evaluate(model,cache,yy,va);sched.step()
        history.append({'epoch':ep+1,'train_ce':total/len(tr),'validation':score,'seconds':round(time.time()-started,1)})
        if score['ce']<best:
            best=score['ce'];stale=0
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'best_epoch':ep+1,'validation':score,'counts':COUNTS,'resolution':SIDE,'selection':'validation concept CE only'},FB/'model.pt')
        else:stale+=1
        save('training_history.json',history);print(json.dumps(history[-1]),flush=True)
        if stale>=4:break
    del model;torch.cuda.empty_cache()


def model_load():
    data=torch.load(FB/'model.pt',map_location='cpu',weights_only=False)
    model=ImageConceptModel(count=sum(COUNTS));model.load_state_dict(data['state_dict']);return model.cuda().eval()


def predictions():
    if (FB/'predictions.npz').exists():return
    model=model_load();saved={}
    with torch.inference_mode():
        for split in ['train','test']:
            cache=np.load(FB/(split+'_rgb.npy'),mmap_mode='r');pp=[];ff=[];zz=[]
            for start in range(0,len(cache),192):
                z,f,_=model(tensor(cache,np.arange(start,min(start+192,len(cache)))),True)
                pp.append(probability(z).cpu().numpy());ff.append(f.cpu().numpy());zz.append(z.cpu().numpy())
            saved[split+'_p']=np.concatenate(pp);saved[split+'_f']=np.concatenate(ff);saved[split+'_z']=np.concatenate(zz)
    np.savez_compressed(FB/'predictions.npz',**saved)
    del model;torch.cuda.empty_cache()


def fit_readout(values,classes,tr,va):
    configure(0);x=torch.tensor(values,device='cuda');y=torch.tensor(classes,device='cuda')
    head=torch.nn.Linear(x.shape[1],50).cuda();opt=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.01)
    best=(-1.,-1e9);state=None;stale=0
    for ep in range(180):
        order=np.random.permutation(tr)
        for start in range(0,len(order),1024):
            idx=order[start:start+1024];opt.zero_grad(set_to_none=True);loss=F.cross_entropy(head(x[idx]),y[idx]);loss.backward();opt.step()
        with torch.no_grad():
            z=head(x[va]);score=((z.argmax(-1)==y[va]).float().mean().item(),-F.cross_entropy(z,y[va]).item())
        if score>best:best=score;state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};stale=0;epoch=ep+1
        else:stale+=1
        if stale>=25:break
    return {'state_dict':state,'validation_accuracy':best[0],'selected_epoch':epoch}


def readouts():
    if (FB/'readouts.pt').exists():return
    meta=np.load(FB/'metadata.npz');pred=np.load(FB/'predictions.npz')
    valid=(meta['train_labels']>0).all(1);tr=np.flatnonzero(valid&~meta['validation']);va=np.flatnonzero(valid&meta['validation'])
    cbm=fit_readout(pred['train_p'],meta['train_classes'],tr,va)
    mean=pred['train_f'][tr].mean(0);std=pred['train_f'][tr].std(0).clip(.01)
    features=fit_readout(((pred['train_f']-mean)/std).clip(-8,8),meta['train_classes'],tr,va)
    features.update(mean=mean,std=std)
    torch.save({'cbm':cbm,'features':features,'train_complete_images':len(tr),'validation_complete_images':len(va)},FB/'readouts.pt')
    print(json.dumps({'readout_validation':{'cbm':cbm['validation_accuracy'],'features':features['validation_accuracy']}}),flush=True)


def linear_head(record):
    head=torch.nn.Linear(record['state_dict']['weight'].shape[1],50);head.load_state_dict(record['state_dict']);return head.cuda().eval()


def hard(p):
    out=np.zeros_like(p)
    for a,b in zip(OFFSETS[:-1],OFFSETS[1:]):out[np.arange(len(p)),a+p[:,a:b].argmax(-1)]=1
    return out


def alternatives(yy,parts):
    result=yy.copy()
    for i,row in enumerate(yy):
        for j,p in enumerate(PARTS):
            if p in ['tail','wing']:
                old=parts[p][row[j]-1];new=dict(old);new['color']=CYCLE[old['color']];result[i,j]=parts[p].index(new)+1
            else:result[i,j]=1+row[j]%(COUNTS[j]-1)
    return result


def semantic():
    if (FB/'semantic_summary.json').exists():return
    model=model_load();meta=np.load(FB/'metadata.npz');parts=json.loads((FB/'parts.json').read_text(encoding='utf-8'))
    yy=meta['test_labels'];y=meta['test_classes'];pred=np.load(FB/'predictions.npz');records=torch.load(FB/'readouts.pt',weights_only=False,map_location='cpu')
    head=linear_head(records['cbm']);feathead=linear_head(records['features']);p=pred['test_p'];z=pred['test_z']
    group_pred=np.stack([p[:,a:b].argmax(-1) for a,b in zip(OFFSETS[:-1],OFFSETS[1:])],1)
    with torch.no_grad():
        soft_class=head(torch.tensor(p,device='cuda')).argmax(-1).cpu().numpy()
        hard_class=head(torch.tensor(hard(p),device='cuda')).argmax(-1).cpu().numpy()
        fp=((pred['test_f']-records['features']['mean'])/records['features']['std']).clip(-8,8)
        feature_class=feathead(torch.tensor(fp,device='cuda')).argmax(-1).cpu().numpy()
        oracle=np.zeros_like(p)
        for j,a in enumerate(OFFSETS[:-1]):oracle[np.arange(len(p)),a+yy[:,j]]=1
        oracle_class=head(torch.tensor(oracle,device='cuda')).argmax(-1).cpu().numpy()
    result={'test_images':len(p),'group_accuracy':{part:float((group_pred[:,j]==yy[:,j]).mean()) for j,part in enumerate(PARTS)},'all_groups_correct':float((group_pred==yy).all(1).mean()),'class_accuracy':{'soft_cbm':float((soft_class==y).mean()),'hard_same_readout':float((hard_class==y).mean()),'matched_frozen_features':float((feature_class==y).mean()),'oracle_concepts_same_readout':float((oracle_class==y).mean())},'deletions':{},'auxiliary_recolor':{}}
    cache=np.load(FB/'test_rgb.npy',mmap_mode='r');deleted=np.load(FB/'deleted_rgb.npy',mmap_mode='r');masks=np.load(FB/'test_masks.npy')
    all_del=[];recolored={};new_labels={}
    with torch.inference_mode():
        for start in range(0,len(p),64):
            ids=np.arange(start,min(start+64,len(p)))
            x=tensor(deleted,ids).reshape(-1,3,SIDE,SIDE)
            all_del.append(probability(model(x)).cpu().numpy().reshape(len(ids),5,-1))
        dp=np.concatenate(all_del)
        for j in [3,4]:
            pp=[];new=alternatives(yy,parts)[:,j]
            for start in range(0,len(p),64):
                ids=np.arange(start,min(start+64,len(p)));x=tensor(cache,ids);mask=torch.tensor(masks[ids,j,None],device='cuda')
                changed=x*(1-mask)+x.roll(1,1)*mask
                assert ((changed-x)*(1-mask)).abs().max().item()==0
                pp.append(probability(model(changed)).cpu().numpy())
            recolored[PARTS[j]]=np.concatenate(pp);new_labels[PARTS[j]]=new
    for r,part in enumerate(PARTS):
        after=dp[:,r];guess=np.stack([after[:,a:b].argmax(-1) for a,b in zip(OFFSETS[:-1],OFFSETS[1:])],1)
        other=np.arange(5)!=r
        before_cl=torch.from_numpy(p);after_cl=torch.from_numpy(after);w=head.weight.detach().cpu()[y]
        contrib=(before_cl-after_cl)*w
        groups=np.stack([contrib[:,a:b].sum(-1).numpy() for a,b in zip(OFFSETS[:-1],OFFSETS[1:])],1)
        direct=F.linear(before_cl,head.weight.cpu(),head.bias.cpu())-F.linear(after_cl,head.weight.cpu(),head.bias.cpu())
        error=np.abs(contrib.sum(-1).numpy()-direct.detach().numpy()[np.arange(len(y)),y]).max()
        result['deletions'][part]={'target_absent_accuracy':float((guess[:,r]==0).mean()),'target_absent_probability_gain':float((after[:,OFFSETS[r]]-p[:,OFFSETS[r]]).mean()),'other_groups_prediction_unchanged':float((guess[:,other]==group_pred[:,other]).mean()),'other_groups_accuracy':float((guess[:,other]==yy[:,other]).mean()),'class_readout_closure_max_error':float(error),'target_group_fraction_absolute_class_contribution':float((np.abs(groups[:,r])/np.abs(groups).sum(-1).clip(1e-8)).mean())}
    for j in [3,4]:
        part=PARTS[j];after=recolored[part];new=new_labels[part]
        guess=np.stack([after[:,a:b].argmax(-1) for a,b in zip(OFFSETS[:-1],OFFSETS[1:])],1);other=np.arange(5)!=j
        valid=group_pred[:,j]==yy[:,j];success=(guess[:,j]==new)
        details=parts[part];models=[None]+[d['model'] for d in details];colors=[None]+[d['color'] for d in details]
        shape=np.array([models[k] for k in guess[:,j]]);true_shape=np.array([models[k] for k in yy[:,j]])
        color=np.array([colors[k] for k in guess[:,j]]);desired_color=np.array([colors[k] for k in new])
        rows=np.arange(len(p));oldcol=OFFSETS[j]+yy[:,j];newcol=OFFSETS[j]+new
        result['auxiliary_recolor'][part]={'joint_variant_success_all':float(success.mean()),'joint_variant_success_conditional_original_correct':float(success[valid].mean()),'original_correct_coverage':float(valid.mean()),'desired_color_accuracy':float((color==desired_color).mean()),'original_geometry_accuracy_after_edit':float((shape==true_shape).mean()),'other_groups_prediction_unchanged':float((guess[:,other]==group_pred[:,other]).mean()),'other_groups_accuracy':float((guess[:,other]==yy[:,other]).mean()),'new_vs_old_probability_margin_gain':float(((after[rows,newcol]-after[rows,oldcol])-(p[rows,newcol]-p[rows,oldcol])).mean()),'interpretation':'Auxiliary RGB channel-cycle probe inside official part mask; no new rendering, imperfect boundaries, not a gold counterfactual.'}
    np.savez_compressed(FB/'semantic_predictions.npz',deletion_p=dp,**{key+'_recolor_p':value for key,value in recolored.items()})
    save('semantic_summary.json',result);print(json.dumps(result),flush=True)


def contrast_gradients(model,x,queries,alts):
    x=x.detach().requires_grad_(True);logits,features,spatial=model(x,True)
    targets=logits.gather(1,queries)-logits.gather(1,alts);gg=[]
    for q in range(5):gg.append(torch.autograd.grad(targets[:,q].sum(),x,retain_graph=q<4)[0].detach())
    weights=model.head.weight[queries]-model.head.weight[alts]
    cam=F.relu(torch.einsum('bqc,bchw->bqhw',weights,spatial.detach()))
    cam=F.interpolate(cam,size=(SIDE,SIDE),mode='bilinear',align_corners=False)
    return torch.stack(gg,1),targets.detach(),cam.detach()


def normal(v):
    v=v.clamp_min(0);s=v.sum((-2,-1),keepdim=True)
    return torch.where(s>1e-12,v/s.clamp_min(1e-12),torch.full_like(v,1/(SIDE*SIDE)))


def explain():
    if (FB/'explanations.npz').exists():return
    model=model_load()
    for p in model.parameters():p.requires_grad_(False)
    meta=np.load(FB/'metadata.npz');parts=json.loads((FB/'parts.json').read_text(encoding='utf-8'))
    ids=np.concatenate([np.flatnonzero(meta['test_classes']==c)[:4] for c in range(50)])
    cache=np.load(FB/'test_rgb.npy',mmap_mode='r');deleted=np.load(FB/'deleted_rgb.npy',mmap_mode='r');masks=np.load(FB/'test_masks.npy')
    yy=meta['test_labels'];aa=alternatives(yy,parts);qall=yy+OFFSETS[:-1];aall=aa+OFFSETS[:-1]
    saved={'indices':ids,'queries':qall[ids],'alternatives':aall[ids]};collect={};started=time.time()
    original_w=model.head.weight.detach().clone();original_b=model.head.bias.detach().clone()
    generator=torch.Generator(device='cuda');generator.manual_seed(821)
    random_weights=[]
    for seed in range(3):
        w=torch.randn(original_w.shape,device='cuda',generator=generator)
        w=w/w.norm(dim=1,keepdim=True)*original_w.norm(dim=1,keepdim=True);random_weights.append(w)
    def put(key,value):collect.setdefault(key,[]).append(value.detach().cpu().numpy())
    for start in range(0,len(ids),8):
        idx=ids[start:start+8];x=tensor(cache,idx);d=tensor(deleted,idx);mask=torch.tensor(masks[idx],device='cuda',dtype=torch.float32)
        q=torch.tensor(qall[idx],device='cuda');a=torch.tensor(aall[idx],device='cuda')
        g,original,cam=contrast_gradients(model,x,q,a)
        mag=normal(g.abs().sum(2));cam=normal(cam);prior=torch.zeros_like(mag)
        with torch.no_grad():rownorm=model.head.weight.norm(dim=1)
        for w in random_weights:
            with torch.no_grad():model.head.weight.copy_(w);model.head.bias.zero_()
            rg,_,_=contrast_gradients(model,x,q,a);prior+=normal(rg.abs().sum(2))/3
        with torch.no_grad():model.head.weight.copy_(original_w);model.head.bias.copy_(original_b)
        maps={'absolute_gradient':mag,'gradcam':cam,'random_head_prior':prior,'null_subtract':normal((mag-prior).clamp_min(0))}
        area=mask.sum((-2,-1)).clamp_min(1)
        for key,v in maps.items():
            put(key+'__region_density',torch.einsum('bqhw,brhw->bqr',v,mask)/area[:,None])
            put(key+'__region_energy',torch.einsum('bqhw,brhw->bqr',v,mask))
            flat=v.flatten(2).argmax(-1);hit=mask.flatten(2).gather(2,flat[:,:,None]).squeeze(-1)
            put(key+'__own_pointing_hit',hit)
        delta=x[:,None]-d
        put('signed_real_delta',torch.einsum('bqchw,brchw->bqr',g,delta))
        with torch.inference_mode():
            z=model(d.reshape(-1,3,SIDE,SIDE)).reshape(len(x),5,-1)
            after=(z.gather(2,q[:,None].expand(-1,5,-1))-z.gather(2,a[:,None].expand(-1,5,-1))).transpose(1,2)
            put('actual_real_deletion',original[:,:,None]-after)
        for operator in ['blur','mean']:
            if operator=='blur':
                blurred=[]
                for xx in np.asarray(cache[idx]):
                    im=Image.fromarray(xx.transpose(1,2,0));blurred.append(np.asarray(im.filter(ImageFilter.GaussianBlur(6))).copy())
                b=torch.tensor(np.stack(blurred),device='cuda').permute(0,3,1,2).float()/255
            else:b=x.mean((-2,-1),keepdim=True).expand_as(x)
            gd=torch.zeros_like(g)
            for t in (np.arange(8)+.5)/8:
                gi,_,_=contrast_gradients(model,x-float(t)*(x-b),q,a);gd+=gi/8
            local_delta=mask[:,:,None]*(x-b)[:,None]
            put(operator+'__signed_mask_delta',torch.einsum('bqchw,brchw->bqr',g,local_delta))
            put(operator+'__path8_mask_delta',torch.einsum('bqchw,brchw->bqr',gd,local_delta))
            put(operator+'__path8_real_delta',torch.einsum('bqchw,brchw->bqr',gd,delta))
        if start%32==0:print(json.dumps({'explain_images':start+len(idx),'seconds':round(time.time()-started,1)}),flush=True)
    for key,value in collect.items():
        saved[key]=np.concatenate(value);assert np.isfinite(saved[key]).all(),key
    np.savez_compressed(FB/'explanations.npz',**saved)


def summarize():
    data=np.load(FB/'explanations.npz');truth=data['actual_real_deletion'];ids=data['indices'];masks=np.load(FB/'test_masks.npy')[ids]
    present=masks.sum((-2,-1))>0;result={'audit_images':len(ids),'visible_query_coverage':float(present.mean()),'response_rank':{},'localization':{}}
    for key in data.files:
        if key.endswith(('__region_density','__region_energy')):
            corr=correlations(data[key],truth);result['response_rank'][key]=interval(corr)
        elif key.endswith('delta'):
            corr=correlations(data[key],truth);result['response_rank'][key]=interval(corr)
    base=correlations(data['absolute_gradient__region_density'],truth)
    for key in result['response_rank']:
        result['response_rank'][key]['paired_change_from_abs_gradient']=interval(correlations(data[key],truth)-base)
        result['response_rank'][key]['paired_change_from_gradcam_density']=interval(correlations(data[key],truth)-correlations(data['gradcam__region_density'],truth))
    for method in ['absolute_gradient','gradcam','random_head_prior','null_subtract']:
        matrix=data[method+'__region_density'];own=np.diagonal(matrix,axis1=1,axis2=2)
        other=(matrix.sum(-1)-own)/4
        energy=np.diagonal(data[method+'__region_energy'],axis1=1,axis2=2)
        point=data[method+'__own_pointing_hit']
        def masked_mean(x):return np.divide((x*present).sum(1),present.sum(1),out=np.full(len(present),np.nan),where=present.sum(1)>0)
        result['localization'][method]={'own_energy':interval(masked_mean(energy)),'own_pointing_hit':interval(masked_mean(point)),'own_minus_other_region_density':interval(masked_mean(own-other))}
    diag=np.diagonal(truth,axis1=1,axis2=2)
    result['real_deletion_semantics']={'true_vs_alternative_margin_drop_on_target_part':interval(diag.mean(1)),'target_part_response_greater_than_other_mean':interval((diag>(truth.sum(-1)-diag)/4).mean(1)),'positive_target_margin_drop_rate':interval((diag>0).mean(1))}
    result['interpretation']={'generic_map':'Grad-CAM, absolute gradient, random-head subtraction have no access to a deleted image. Mask-delta scores use only original image, chosen blur/mean baseline and masks, so real deletion is a separate response target.','paired_image_predictors':'signed_real_delta and path8_real_delta need the deleted image pixels as input. Their improvement is a directional approximation diagnostic, not evidence a standalone heatmap explains unseen semantic interventions.','no_metric_equivalence':'Raw Spearman, not official SD=(rho+1)/2. Own-pointing and energy are our concept-contrast audit, not full original FunnyBirds CSDC/TS reproduction.'}
    save('explanation_summary.json',result)
    print(json.dumps(result),flush=True)
    return result


def report(summary):
    sem=json.loads((FB/'semantic_summary.json').read_text(encoding='utf-8'));prep=json.loads((FB/'prepared.json').read_text(encoding='utf-8'));rows=[]
    for method,label in [('absolute_gradient__region_density','绝对梯度区域密度'),('null_subtract__region_density','随机头先验相减'),('gradcam__region_density','Grad-CAM区域密度'),('absolute_gradient__region_energy','绝对梯度区域总量'),('gradcam__region_energy','Grad-CAM区域总量'),('blur__signed_mask_delta','Taylor，模糊掩码方向'),('blur__path8_mask_delta','8步全图路径，模糊掩码方向'),('mean__signed_mask_delta','Taylor，均值掩码方向'),('mean__path8_mask_delta','8步全图路径，均值掩码方向'),('signed_real_delta','已知删除图的Taylor【条件化诊断】'),('blur__path8_real_delta','已知删除图的模糊路径【条件化诊断】'),('mean__path8_real_delta','已知删除图的均值路径【条件化诊断】')]:
        item=summary['response_rank'][method];ci=item['ci95'];rows.append(f"| {label} | {item['mean']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |")
    table='\n'.join(rows);accuracy=sem['class_accuracy']
    deletion='\n'.join(f"| {part} | {100*sem['group_accuracy'][part]:.1f}% | {100*sem['deletions'][part]['target_absent_accuracy']:.1f}% | {100*sem['deletions'][part]['other_groups_prediction_unchanged']:.1f}% |" for part in PARTS)
    recolor='\n'.join(f"| {part} | {100*v['joint_variant_success_all']:.1f}% | {100*v['desired_color_accuracy']:.1f}% | {100*v['original_geometry_accuracy_after_edit']:.1f}% | {100*v['other_groups_prediction_unchanged']:.1f}% |" for part,v in sem['auxiliary_recolor'].items())
    text=f'''# FunnyBirds：语义闭环与解释响应先导实验

数据来源：用户提供main_python/dataset/fbdl/FunnyBirds.zip；原zip不改动，直接读取，不重复解压12万文件。训练/验证/测试=45000/5000/500；验证从官方训练集中固定分层划出。原测试集在模型选定后使用。输入160×160，单种子0，ResNet18预训练；本轮不是官方全协议复现或新方法主张。

## 训练与标注核查

训练标签逐图读取实际部位model/color；约四分之一的部位已经被删除，显式加入“缺失”类，不能照抄源类别的完整属性。五组共31个输出（26个存在变体+5个缺失类别）。概念编码器只优化概念交叉熵，不用类别损失、不用位置损失，不输入分割图。模型仅按验证概念CE选择。

冻结概念后，学习线性50类读出；只用完整训练图训练读出，避免给被删成同一外观的图强贴不同源类别。匹配特征读出使用同一批完整训练图。分割与原生删除图只用于测试。原生保留全部部位重渲染与原图的最大像素差为{prep['maximum_difference_full_kept_render_vs_original_uint8']:.1f}/255，已核查删除文件命名是保留部位集合。

## 500张官方测试图的分类

- 软概念CBM：{accuracy['soft_cbm']:.1%}。
- 相同读出换为硬概念：{accuracy['hard_same_readout']:.1%}。
- 同一冻结骨干的特征线性读出：{accuracy['matched_frozen_features']:.1%}。
- 真实概念输入同一CBM读出：{accuracy['oracle_concepts_same_readout']:.1%}（上限诊断，不是视觉预测成绩）。
- 五组概念同时判断正确：{sem['all_groups_correct']:.1%}。

## 原生删除：能否识别目标缺失并保留其他概念

| 部位 | 原图变体准确率 | 删除后识别该部位缺失 | 其他四组预测保持率 |
|---|---:|---:|---:|
{deletion}

这些是官方现成的同场景删除图，不是灰块遮挡。删除后类别可能不唯一，不能直接沿用原类别计算“删除图分类准确率”。线性读出贡献求和可重建类别logit变化，其代数闭合本身不是语义正确证明。

## 同部位属性：辅助换色探针

对tail/wing沿RGB通道循环，使红→绿→蓝→红；只编辑官方部位掩码，保持几何与外部像素。期望目标为相同geometry、下一种color的概念类别。此操作在线生成，仅作探针，不建立新数据集或新标注；**它不是官方重渲染换色反事实**，抗锯齿边界不完美。

| 部位 | 目标新变体准确率（含几何与色彩） | 目标新颜色准确率 | 原几何准确率 | 其他四组预测保持率 |
|---|---:|---:|---:|---:|
{recolor}

完整结果还保留原概念正确条件下的换色成功率及覆盖率；不只汇报条件成功。

## 200张固定解释子集：实际删除响应相关性

每类预先取前4张，共200图，每图5个“真实变体−同部位另一变体”对比目标；tail/wing对照保持geometry、改变color。比较5个真实删除部位后的响应排序，按图像bootstrap。不是官方将Spearman映射到[0,1]的SD分数。

| 方法 | 平均Spearman | 95%区间 |
|---|---:|---|
{table}

必须区分两类：通用归因与掩码方向评分不读取删除图，真实删除是独立评估目标；“已知删除图”的方法读取像素差，只验证一阶/路径近似能力，不能当成独立解释优越性。随机头是保留训练骨干、仅换线性头的3个阴性参照，不能称为纯架构效应。

定位能量、最大点命中、同部位选择性，以及真实目标变体相对替代变体在删除下的响应，均完整保存在explanation_summary.json；没有按模型预测正确筛掉困难样本。

## 可复现文件

- 脚本：main_python/concept_null_audit/funnybirds.py。
- 本轮目录：main_python/concept_null_audit/artifacts/funnybirds_2026-11-15。
- 固定协议protocol.json，训练日志training_history.json，模型model.pt，读出readouts.pt。
- semantic_summary.json及explanation_summary.json保存全部指标和限制。
- 执行：工作目录main_python，python -X utf8 -B -u -m concept_null_audit.funnybirds --stage all。

依据：[官方框架](https://github.com/visinf/funnybirds-framework)、[官方标签和部位解码](https://github.com/visinf/funnybirds-framework/blob/main/datasets/funny_birds.py)、[原生删除评估](https://github.com/visinf/funnybirds-framework/blob/main/evaluation_protocols.py)。本轮改变分辨率和模型/概念结构，没有冒充官方原论文性能复现。
'''
    (BASE/'FunnyBirds语义与解释验证.md').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['all','prepare','train','evaluate','report'],default='all');parser.add_argument('--epochs',type=int,default=12);args=parser.parse_args()
    configure(0);FB.mkdir(parents=True,exist_ok=True);ast.parse(Path(__file__).read_text(encoding='utf-8'))
    if args.stage in ['all','prepare']:prepare()
    if args.stage in ['all','train']:train(args.epochs);predictions();readouts()
    if args.stage in ['all','evaluate']:semantic();explain()
    if args.stage in ['all','evaluate','report']:
        summary=summarize();report(summary)
        save('verification.json',{'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_sha256':hashlib.sha256((FB/'model.pt').read_bytes()).hexdigest(),'zip_unmodified':'Source zip opened read-only only; no archive extraction or deletion.','all_outputs_inside':str(FB),'stage':args.stage,'test_used_for_model_selection':False})
        print(BASE/'FunnyBirds语义与解释验证.md',flush=True)


if __name__=='__main__':main()
