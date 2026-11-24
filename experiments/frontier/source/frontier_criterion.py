"""Test whether existing localization contrasts predict renderer responses."""
from .common import ROOT, OUT, write_json
from .frontier_protocol import DEST
import json
import numpy as np
from scipy.stats import rankdata
import torch


def row_correlation(x, y, axis=1):
    x = rankdata(x, axis=axis); y = rankdata(y, axis=axis)
    x = x - x.mean(axis=axis, keepdims=True)
    y = y - y.mean(axis=axis, keepdims=True)
    scale = np.sqrt((x*x).sum(axis=axis) * (y*y).sum(axis=axis))
    return np.divide((x*y).sum(axis=axis), scale, out=np.full_like(scale, np.nan), where=scale > 0)


def interval(values):
    values = np.asarray(values)
    return np.quantile(values[np.isfinite(values)], [.025, .975]).tolist()


def hit_scores(context, peaks):
    side = int(context['side'])
    windows = np.unpackbits(context['windows_packed'], axis=1, count=side*side).astype(bool)
    own = windows[np.arange(len(context['image']))[None], peaks[..., 0]*side+peaks[..., 1]].astype(float)
    transferred = peaks[:, context['donors']]
    donor = windows[np.arange(len(context['image']))[None, None], transferred[..., 0]*side+transferred[..., 1]].mean(1)
    return own, own-donor


def main():
    root = ROOT / 'github_cun/main/data/core_audit/data'
    fb = OUT / 'funnybirds_2026-11-15'
    context = dict(np.load(root / 'FB_mask_context.npz'))
    sample = dict(np.load(fb / 'explanations.npz'))
    population = dict(np.load(OUT / 'controlled_audit_2026-11-19/stage2_population.npz'))
    assert np.array_equal(sample['indices'], population['ids'])
    assert np.array_equal(context['image'], np.repeat(np.arange(200), 5))
    assert np.array_equal(context['group'], np.tile(np.arange(5), 200))
    weights = context['weights'].astype(int)
    draws = np.stack([np.repeat(np.arange(200), row) for row in weights])
    truth = sample['actual_real_deletion'].astype(float)
    response = np.diagonal(truth, axis1=1, axis2=2)
    contrast = response - (truth.sum(-1)-response)/4
    scores = {}
    for name in ['cam', 'gradient']:
        arrays = dict(np.load(root / f'FB_mask_{name}.npz'))
        own, gain = hit_scores(context, arrays['peaks'])
        scores[name+'_S'] = own[0].reshape(200, 5)
        scores[name+'_G'] = gain[0].reshape(200, 5)
        scores[name+'_deltaG'] = (gain[0]-gain[1:].mean(0)).reshape(200, 5)
    results = {'scope': 'Exploratory R18 seed0, 200 fixed images, real renderer removals; response association, not ACE or semantic intervention.',
               'primary': {}, 'contrast_response': {}, 'decision_probe': {}}
    boots = {}
    for target_name, target in [('primary', response), ('contrast_response', contrast)]:
        for name, value in scores.items():
            per_part = row_correlation(value, target, axis=0)
            all_draws = row_correlation(value[draws], target[draws], axis=1)
            macro = np.nanmean(all_draws, axis=1)
            boots[target_name+'__'+name] = macro
            results[target_name][name] = {'macro_spearman': float(np.nanmean(per_part)), 'ci95': interval(macro),
                'per_part_rho': per_part.tolist(), 'bootstrap_valid_fraction': float(np.isfinite(macro).mean())}
            print(target_name, name, results[target_name][name]['macro_spearman'], flush=True)
        results[target_name]['deltaG_minus_S'] = {
            name: {'change': results[target_name][name+'_deltaG']['macro_spearman']-results[target_name][name+'_S']['macro_spearman'],
                   'ci95': interval(boots[target_name+'__'+name+'_deltaG']-boots[target_name+'__'+name+'_S'])}
            for name in ['cam', 'gradient']}

    predictions = np.load(fb / 'predictions.npz')
    semantic = np.load(fb / 'semantic_predictions.npz')
    meta = np.load(fb / 'metadata.npz')
    readout = torch.load(fb / 'readouts.pt', map_location='cpu', weights_only=False)['cbm']['state_dict']
    W, b = readout['weight'].numpy(), readout['bias'].numpy()
    p = predictions['test_p'][sample['indices']]
    pd = semantic['deletion_p'][sample['indices']]
    labels = meta['test_classes'][sample['indices']]
    logits = p @ W.T+b
    foil_scores = logits.copy(); foil_scores[np.arange(200), labels] = -np.inf
    foils = foil_scores.argmax(1)
    contrast_weight = W[labels]-W[foils]
    class_response = np.einsum('bc,bkc->bk', contrast_weight, p[:, None]-pd)
    oracle, uniform = class_response.max(1), class_response.mean(1)
    for name, value in scores.items():
        tie = np.isclose(value, value.max(1, keepdims=True), atol=1e-12, rtol=0)
        selected = (class_response*tie).sum(1)/tie.sum(1)
        gain = selected-uniform
        regret = oracle-selected
        results['decision_probe'][name] = {
            'selected_minus_uniform_margin_response': float(gain.mean()), 'ci95': interval(gain[draws].mean(1)),
            'oracle_regret': float(regret.mean()), 'regret_ci95': interval(regret[draws].mean(1)),
            'mean_tie_count': float(tie.sum(1).mean())}
    results['decision_probe']['uniform_oracle_regret'] = float((oracle-uniform).mean())
    results['scope_limits'] = ['Class-margin task is not the concept-margin query used to generate maps.',
                              'No held-out calibration, human decision utility, or universal causal claim.',
                              'Intervals condition on fixed models, finite references and donors; score selection is exploratory.']
    np.savez_compressed(DEST / 'criterion_inputs.npz', **scores, concept_response=response,
                        specific_response=contrast, class_response=class_response, weights=weights, ids=sample['indices'])
    write_json(DEST / 'criterion_results.json', results)
    print(DEST / 'criterion_results.json')


if __name__ == '__main__':
    main()
