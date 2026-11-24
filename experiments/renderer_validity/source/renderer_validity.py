"""Extend the criterion check without training or fabricating semantic edits.

Usage from main_python: python -B -m concept_null_audit.renderer_validity
The 300 follow-up images previously supplied layout calibration. They are a
complementary localization cohort, NOT pristine held-out confirmation.
"""
from .common import ROOT, OUT, configure, write_json
from .models import ImageConceptModel
from .funnybirds import alternatives, OFFSETS
from .renderer_statistics import analyze
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import distance_transform_cdt

RUN = OUT/'renderer_validity_2026-11-24'
FB = OUT/'funnybirds_2026-11-15'
SIDE = 160
MODELS = [('resnet18_seed0', 'model.pt', 'resnet18'),
          ('resnet18_seed1', 'seed1_model.pt', 'resnet18'),
          ('resnet50_seed2', 'r50_model.pt', 'resnet50')]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    RUN.mkdir(exist_ok=True)
    value = {
        'status': 'Exploratory follow-up; original 200-image outcomes and all-model aggregate quality known. Plan written before this follow-up inference.',
        'cohorts': {'original200': 'First four images/class, unchanged; implementation check and descriptive replication.',
                    'complement300': 'Remaining six images/class; previously used for layout calibration, not independent confirmation.'},
        'models': [v[0] for v in MODELS],
        'intervention': 'Official same-scene render removing exactly the queried part; fixed camera, scene and remaining parts. Includes occlusion/shadow changes. No invented recolor/scale counterfactuals.',
        'outcome': 'Signed decrease of the original ground-truth concept-minus-fixed-alternative logit margin. No correctness filter. A model outcome under this operation, not natural-world semantic ACE.',
        'secondary_outcome': 'Own-part concept-margin response minus the mean of the other four part-deletion responses.',
        'scores': 'Per-query S, G and deltaG at existing mask dilation r24; same contrast target, trained and three row-norm-matched Gaussian CUDA head draws seed821; fixed heads.',
        'donors': '200 same-(query,alternative) derangements inside each cohort. Original schedule reused exactly. Complement schedule seed2026112405, never align images or use outcome values.',
        'support': 'Primary: all queries. Sensitivity: recipients with at least one decoded visible part pixel at160px. Donors unchanged in sensitivity. All 500 ground-truth parts are present; empty masks can be occluded/thin and are not absent concepts.',
        'primary_analysis': 'Equal-part mean of five within-part Spearman correlations; paired image-cluster bootstrap,3000 draws, fixed heads/donors. Compare deltaG minus S for each of3models x2explainers, with approximate simultaneous max-standardized intervals over six comparisons, separately for each cohort/support. No omnibus familywise claim over secondary analyses.',
        'secondary_utility': 'Only original R18s0 has a frozen class readout. Rank five parts by each score, average ties, measure true-class/original-best-rival logit-margin deletion response, uniform choice and oracle regret. No training selector or human-utility claim; map target and utility outcome differ.',
        'zero_variance': 'A constant score/outcome yields undefined correlation; do not silently average fewer parts. Report valid bootstrap fraction.',
        'bootstrap_seeds': {'original200': 'Reuse published exact image counts.', 'complement300': 2026112406},
        'limits': ['No new training seeds; three already selected checkpoints.', 'No CUB inpainting/ACE claim.', 'No new explanation metric or priority claim.', 'CI crossing zero does not imply equivalence or absence.', 'Support analyses change recipient populations, not a decomposition of the original association.'],
        'inputs': {v[1]: digest(FB/v[1]) for v in MODELS},
    }
    path = RUN/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == value, 'Protocol drift'
    else:
        write_json(path, value)
    return value


def schedules(q, a, seed):
    rng = np.random.default_rng(seed)
    n = len(q)
    donor = np.broadcast_to(np.arange(n)[None, :, None], (200, n, 5)).copy()
    for part in range(5):
        keys = np.stack([q[:, part], a[:, part]], 1)
        for key in np.unique(keys, axis=0):
            members = np.flatnonzero((keys == key).all(1))
            assert len(members) > 1, 'No singleton removal is allowed silently.'
            for permutation in range(200):
                order = rng.permutation(members)
                donor[permutation, order, part] = np.roll(order, int(rng.integers(1, len(order))))
    assert (donor != np.arange(n)[None, :, None]).all()
    assert np.array_equal(q[donor, np.arange(5)], np.broadcast_to(q, donor.shape))
    assert np.array_equal(a[donor, np.arange(5)], np.broadcast_to(a, donor.shape))
    return donor.astype(np.int16)


def peak(raw):
    sums = raw.sum((-2, -1), keepdim=True)
    normalized = torch.where(sums > 1e-12, raw/sums.clamp_min(1e-12),
                             torch.full_like(raw, 1/(SIDE*SIDE)))
    index = normalized.flatten(2).argmax(-1).cpu().numpy()
    return np.stack([index//SIDE, index % SIDE], -1).astype(np.int16)


def paired_responses(model, rgb, deleted, order):
    """Evaluate original plus five interventions in one shared forward batch.

    Identical no-op renderings must not acquire response ranks from evaluating
    original/intervened images with different convolution batch shapes.
    """
    logits = np.empty((500, 31), np.float32)
    removed = np.empty((500, 5, 31), np.float32)
    with torch.inference_mode():
        for start in range(0, len(order), 8):
            rows = order[start:start+8]
            combined = np.concatenate([np.array(rgb[rows])[:, None],
                                       np.array(deleted[rows])], axis=1)
            x = torch.as_tensor(combined, device='cuda').float().div_(255)
            z = model(x.flatten(0, 1)).reshape(len(rows), 6, 31).cpu().numpy()
            logits[rows], removed[rows] = z[:, 0], z[:, 1:]
    return logits, removed


def infer():
    protocol = prepare()
    configure(0)
    meta = dict(np.load(FB/'metadata.npz'))
    labels = meta['test_labels']
    assert not (labels == 0).any(), 'The currently declared complete-part test population changed.'
    query = labels+OFFSETS[:-1]
    alternative = alternatives(labels, json.loads((FB/'parts.json').read_text()))+OFFSETS[:-1]
    original = np.load(FB/'explanations.npz')['indices']
    complement = np.setdiff1d(np.arange(len(labels)), original)
    order = np.r_[original, complement]
    assert len(original) == 200 and len(complement) == 300
    rgb = np.load(FB/'test_rgb.npy', mmap_mode='r')
    deleted = np.load(FB/'deleted_rgb.npy', mmap_mode='r')
    masks = np.load(FB/'test_masks.npy', mmap_mode='r').astype(bool)
    prepared = json.loads((FB/'prepared.json').read_text())
    assert prepared['maximum_difference_full_kept_render_vs_original_uint8'] == 0
    inputs = {'queries': query, 'alternatives': alternative, 'classes': meta['test_classes'],
              'visible': masks.sum((-2, -1)) > 0, 'side': np.array(SIDE),
              'original200_rows': original, 'complement300_rows': complement}
    pixels = np.abs(rgb[:, None].astype(np.int16)-deleted.astype(np.int16))
    inputs['unchanged'] = pixels.max((-3, -2, -1)) == 0
    inputs['pixel_mean_l1'] = pixels.mean((-3, -2, -1))/255
    del pixels
    windows = np.empty_like(masks)
    for i in range(len(masks)):
        for j in range(5):
            windows[i, j] = (distance_transform_cdt(~masks[i, j], metric='chessboard') <= 24
                             if masks[i, j].any() else np.zeros((SIDE, SIDE), bool))
    inputs['windows_packed'] = np.packbits(windows.reshape(500, 5, -1), axis=-1)
    context = np.load(ROOT/'github_cun/main/data/core_audit/data/FB_mask_context.npz')
    old = context['donors'].reshape(200, 200, 5)
    inputs['original200_donors'] = (old//5).astype(np.int16)
    assert np.array_equal(schedules(query[original], alternative[original], 2026111901), inputs['original200_donors'])
    inputs['complement300_donors'] = schedules(query[complement], alternative[complement], 2026112405)
    inputs['original200_weights'] = context['weights'].astype(np.int16)
    inputs['complement300_weights'] = np.random.default_rng(2026112406).multinomial(
        300, np.full(300, 1/300), size=3000).astype(np.int16)
    checks = {'same_scene_full_render_exact': True, 'all_test_parts_annotated_present': True,
              'support_is_visibility_not_presence': True, 'checkpoints': {}, 'historical_matches': {}}
    peak_arrays, logits_arrays, deletion_arrays = [], [], []
    for label, checkpoint, architecture in MODELS:
        target = RUN/(label+'.npz')
        torch.backends.cudnn.allow_tf32 = label == 'resnet18_seed0'
        if target.exists():
            cache = dict(np.load(target))
            if cache.get('response_contract', np.array('')).item() != 'paired_six_same_batch_v1':
                model = ImageConceptModel(architecture, count=31).cuda().eval()
                model.load_state_dict(torch.load(FB/checkpoint, map_location='cpu', weights_only=False)['state_dict'])
                cache['logits'], cache['deleted_logits'] = paired_responses(model, rgb, deleted, order)
                cache['response_contract'] = np.array('paired_six_same_batch_v1')
                np.savez_compressed(target, **cache)
                del model
                torch.cuda.empty_cache()
            peak_arrays.append(cache['peaks']); logits_arrays.append(cache['logits'])
            deletion_arrays.append(cache['deleted_logits'])
            print('using completed inference', label, flush=True)
            continue
        started = time.time()
        model = ImageConceptModel(architecture, count=31).cuda().eval()
        saved = torch.load(FB/checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        trained = model.head.weight.detach().clone()
        bias = model.head.bias.detach().clone()
        generator = torch.Generator(device='cuda').manual_seed(821)
        head_weights = [trained]
        for _ in range(3):
            w = torch.randn(trained.shape, device='cuda', generator=generator)
            head_weights.append(w/w.norm(dim=1, keepdim=True)*trained.norm(dim=1, keepdim=True))
        all_peaks = np.empty((2, 4, 500, 5, 2), np.int16)
        all_logits = np.empty((500, 31), np.float32)
        all_deleted = np.empty((500, 5, 31), np.float32)
        for start in range(0, 500, 8):
            rows = order[start:start+8]
            q = torch.as_tensor(query[rows], device='cuda')
            a = torch.as_tensor(alternative[rows], device='cuda')
            for head, weight in enumerate(head_weights):
                with torch.no_grad():
                    model.head.weight.copy_(weight)
                    model.head.bias.copy_(bias if head == 0 else torch.zeros_like(bias))
                x = torch.as_tensor(np.array(rgb[rows]), device='cuda').float().div_(255).requires_grad_(True)
                z, _, spatial = model(x, True)
                margin = z.gather(1, q)-z.gather(1, a)
                gradients = torch.stack([torch.autograd.grad(margin[:, j].sum(), x, retain_graph=j < 4)[0].detach()
                                          for j in range(5)], 1).abs().sum(2)
                cam = F.interpolate(torch.einsum('bqc,bchw->bqhw', weight[q]-weight[a], spatial.detach()).relu(),
                                     (SIDE, SIDE), mode='bilinear', align_corners=False)
                all_peaks[0, head, rows] = peak(cam)
                all_peaks[1, head, rows] = peak(gradients)
                if head == 0:
                    all_logits[rows] = z.detach().cpu().numpy()
                    with torch.inference_mode():
                        d = torch.as_tensor(np.array(deleted[rows]), device='cuda').float().div_(255)
                        all_deleted[rows] = model(d.flatten(0, 1)).reshape(len(rows), 5, 31).cpu().numpy()
            if start % 64 == 0 or start+8 >= 500:
                print(json.dumps({'model': label, 'images': min(start+8, 500),
                                  'seconds': round(time.time()-started, 1)}), flush=True)
        with torch.no_grad():
            model.head.weight.copy_(trained); model.head.bias.copy_(bias)
        all_logits, all_deleted = paired_responses(model, rgb, deleted, order)
        np.savez_compressed(target, peaks=all_peaks, logits=all_logits, deleted_logits=all_deleted,
                            response_contract=np.array('paired_six_same_batch_v1'))
        peak_arrays.append(all_peaks); logits_arrays.append(all_logits); deletion_arrays.append(all_deleted)
        del model
        torch.cuda.empty_cache()
    inputs['peaks'] = np.stack(peak_arrays)
    inputs['logits'] = np.stack(logits_arrays)
    inputs['deleted_logits'] = np.stack(deletion_arrays)
    noop_logit_errors = np.abs(inputs['logits'][:, :, None]-inputs['deleted_logits'])[:, inputs['unchanged']]
    checks['maximum_noop_logit_error'] = float(noop_logit_errors.max())
    assert checks['maximum_noop_logit_error'] == 0, 'Identical renderings must have exactly identical outputs.'
    checks['response_contract'] = 'Original and five deletions in one forward batch; float64 contrast subtraction from float32 logits.'
    readout = torch.load(FB/'readouts.pt', map_location='cpu', weights_only=False)['cbm']['state_dict']
    inputs['readout_weight'] = readout['weight'].numpy()
    inputs['readout_bias'] = readout['bias'].numpy()
    for i, (label, checkpoint, _) in enumerate(MODELS):
        reference = np.load(OUT/'controlled_audit_2026-11-19'/('peaks_'+label+'.npz'))
        mismatches = {method: int(np.any(inputs['peaks'][i, k, 0, original] != reference[name], axis=-1).sum())
                       for k, (method, name) in enumerate([('cam', 'gradcam'), ('gradient', 'absolute_gradient')])}
        assert not any(mismatches.values()), (label, mismatches)
        checks['historical_matches'][label] = mismatches
        checks['checkpoints'][label] = protocol['inputs'][checkpoint]
    for k, method in enumerate(['cam', 'gradient']):
        base = np.load(ROOT/f'github_cun/main/data/core_audit/data/FB_mask_{method}.npz')['peaks']
        actual = inputs['peaks'][0, k, :, original].transpose(1, 0, 2, 3).reshape(4, 1000, 2)
        assert np.array_equal(actual, base), ('reference peaks do not reproduce', method)
    np.savez_compressed(RUN/'inputs.npz', **inputs)
    checks['inputs_sha256'] = digest(RUN/'inputs.npz')
    write_json(RUN/'inference_checks.json', checks)
    print('Saved compact criterion inputs', RUN/'inputs.npz', flush=True)


def evaluate():
    data = dict(np.load(RUN/'inputs.npz'))
    result = analyze(data)
    old = json.loads((OUT/'frontier_pilot_2026-11-24/criterion_results.json').read_text())
    new = result['cohorts']['original200']['analyses']['all']['models']['resnet18_seed0']
    errors = []
    # Old responses were saved after float32 subtraction, so require agreement
    # in ranked results rather than claiming bitwise identical raw logits.
    for method in ['cam', 'gradient']:
        for score in ['S', 'G', 'deltaG']:
            observed = new[method]['concept_margin'][score]
            expected = old['primary'][method+'_'+score]
            error = abs(observed['estimate']-expected['macro_spearman'])
            errors.append(error)
            assert error < .01, (method, score, error)
    write_json(RUN/'results.json', result)
    write_json(RUN/'analysis_checks.json', {'original_point_max_error': max(errors),
                'response_rounding': 'Original/deletions evaluated in one shared batch, with no-op output equality; float64 contrasts from float32 logits. Historical subtraction used separate batches and float32.',
                'status': 'passed'})
    for cohort, entry in result['cohorts'].items():
        for support, analysis in entry['analyses'].items():
            for model, methods in analysis['models'].items():
                for method, record in methods.items():
                    values = record['concept_margin']
                    print(cohort, support, model, method,
                          'rho S/deltaG', values['S']['estimate'], values['deltaG']['estimate'],
                          'difference', values['deltaG_minus_S']['estimate'],
                          'valid draws', values['deltaG_minus_S']['valid_bootstrap_fraction'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'infer', 'evaluate', 'all'], default='all')
    args = parser.parse_args()
    if args.stage in ['prepare', 'all']:
        prepare()
    if args.stage in ['infer', 'all']:
        infer()
    if args.stage in ['evaluate', 'all']:
        evaluate()


if __name__ == '__main__':
    main()
