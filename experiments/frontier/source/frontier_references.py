"""Finite-reference ordering gate, with matched paired method contrasts."""
from .common import OUT, configure, write_json
from .frontier_protocol import DEST
from .frontier_criterion import hit_scores
from .models import ImageConceptModel
import numpy as np
import torch
from torch.nn import functional as F


def infer_channel_bank():
    path = DEST / 'fb_channel_peaks.npz'
    if path.exists():
        return dict(np.load(path))
    configure(0)
    fb = OUT / 'funnybirds_2026-11-15'
    pop = dict(np.load(OUT / 'controlled_audit_2026-11-19/stage2_population.npz'))
    model = ImageConceptModel(count=31).cuda().eval()
    model.load_state_dict(torch.load(fb / 'model.pt', map_location='cpu', weights_only=False)['state_dict'])
    for p in model.parameters():
        p.requires_grad_(False)
    trained = model.head.weight.detach().clone()
    rgb = np.load(fb / 'test_rgb.npy', mmap_mode='r')
    output = {}
    permutations = []
    for h, seed in enumerate([2026112401, 2026112402, 2026112403]):
        gen = torch.Generator(device='cuda').manual_seed(seed)
        perm = torch.randperm(trained.shape[1], generator=gen, device='cuda')
        W = trained[:, perm]
        assert torch.allclose(W@W.T, trained@trained.T, rtol=1e-4, atol=1e-5)
        permutations.append(perm.cpu().numpy())
        with torch.no_grad():
            model.head.weight.copy_(W); model.head.bias.zero_()
        caches = {'cam': [], 'gradient': []}
        for start in range(0, 200, 8):
            ids = pop['ids'][start:start+8]
            q = torch.as_tensor(pop['queries'][start:start+8], device='cuda')
            a = torch.as_tensor(pop['alternatives'][start:start+8], device='cuda')
            x = torch.as_tensor(np.array(rgb[ids]), device='cuda').float().div_(255).requires_grad_(True)
            z, _, spatial = model(x, True)
            value = z.gather(1, q)-z.gather(1, a)
            gradients = torch.stack([torch.autograd.grad(value[:, j].sum(), x, retain_graph=j<4)[0].detach().abs().sum(1) for j in range(5)], 1)
            cam = F.interpolate(torch.einsum('bqc,bchw->bqhw', W[q]-W[a], spatial.detach()).relu(), (160,160), mode='bilinear', align_corners=False)
            for name, maps in [('cam', cam), ('gradient', gradients)]:
                index = maps.flatten(2).argmax(-1).cpu().numpy()
                caches[name].append(np.stack([index//160, index%160], -1).astype(np.int16))
        for name, values in caches.items():
            output[f'head{h}__{name}'] = np.concatenate(values)
        print('channel bank head', h, flush=True)
    output['permutations'] = np.stack(permutations)
    np.savez_compressed(path, **output)
    return output


def main():
    from .common import ROOT
    context = dict(np.load(ROOT / 'github_cun/main/data/core_audit/data/FB_mask_context.npz'))
    banks = {}
    for seed in [821, 2026111909, 2026111910]:
        banks[f'gaussian_{seed}'] = dict(np.load(OUT / f'head_robustness_2026-11-19/fb_{seed}.npz'))
    banks['channel'] = infer_channel_bank()
    methods = {name: dict(np.load(ROOT / f'github_cun/main/data/core_audit/data/FB_mask_{name}.npz')) for name in ['cam','gradient']}
    weights = context['weights'].astype(float)
    reference_points, all_differences, bootstrap_differences, saved_indices = {}, [], [], []
    trained = {}
    for name in methods:
        _, gain = hit_scores(context, methods[name]['peaks'][:1])
        trained[name] = gain.reshape(200,5)
    for j, (label, bank) in enumerate(banks.items()):
        per_method = {}
        head_indices = np.random.default_rng(2026112404+j).integers(3, size=(len(weights),3))
        saved_indices.append(head_indices)
        for name in methods:
            peaks = np.stack([bank[f'head{h}__{name}'].reshape(1000,2) for h in range(3)])
            _, gains = hit_scores(context, peaks)
            per_image = gains.reshape(3,200,5).mean(-1).T
            reference_boot = weights@per_image/200
            ref_draw = reference_boot[np.arange(len(weights))[:, None], head_indices].mean(1)
            raw_draw = weights@trained[name].mean(1)/200
            per_method[name] = {'point': float(trained[name].mean()-gains.mean()), 'draws': raw_draw-ref_draw}
        difference = per_method['cam']['point']-per_method['gradient']['point']
        draws = per_method['cam']['draws']-per_method['gradient']['draws']
        reference_points[label] = {name: per_method[name]['point'] for name in per_method}
        reference_points[label].update(cam_minus_gradient=difference, paired_ci95=np.quantile(draws,[.025,.975]).tolist())
        all_differences.append(difference); bootstrap_differences.append(draws)
    points = np.array(all_differences)
    boot = np.stack(bootstrap_differences,1)
    std = boot.std(0,ddof=1)
    quantile = np.quantile(np.max(np.abs((boot-points)/std),axis=1),.95)
    simultaneous = np.stack([points-quantile*std, points+quantile*std],1)
    for label, bounds in zip(banks,simultaneous): reference_points[label]['simultaneous_band']=bounds.tolist()
    # A shared additive reference cannot change a pairwise order.
    A,B = .13,.15
    references = np.array([-.2, 0, .1, .4])
    cancellation_error = float(np.abs(((A-references)-(B-references))-(A-B)).max())
    results = {'scope': 'Exploratory finite four-bank sensitivity, same R18, mask r24, methods/queries/donors fixed.',
               'references': reference_points,
               'point_envelope': [float(points.min()),float(points.max())],
               'simultaneous_envelope': [float(simultaneous[:,0].min()),float(simultaneous[:,1].max())],
               'dominance_resolved_over_declared_banks': bool((simultaneous[:,0]>0).all() or (simultaneous[:,1]<0).all()),
               'common_reference_cancellation_max_error': cancellation_error,
               'not_claimed': ['Sharp bounds over all admissible references','Stochastic dominance from counting positive references','A new robust-ranking theory','Cross-model semantic trustworthiness ranking']}
    np.savez_compressed(DEST / 'reference_order_draws.npz', point=points, draws=boot, simultaneous=simultaneous,head_indices=np.stack(saved_indices))
    write_json(DEST / 'reference_order_results.json',results)
    print(results)


if __name__=='__main__':
    main()
