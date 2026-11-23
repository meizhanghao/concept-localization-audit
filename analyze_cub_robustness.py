"""Paired window comparisons and anatomical influence on the fixed CUB audit.

Exploratory analysis of existing caches. All six width contrasts and every
anatomical group are reported; the primary h=45 population is never selected.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np


WIDTHS = (16, 24, 36, 45)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def interval(draws):
    return np.quantile(draws, [.025, .975], axis=0).T.tolist()


def simultaneous(estimates, draws):
    """Approximate 95% centered-bootstrap max-standardized-deviation band.

    One bootstrap standard deviation per contrast scales the joint draws.
    This is a fixed-family, exploratory band, not selective inference or a
    studentized bootstrap with a separately fitted SE inside every draw.
    """
    estimates = np.asarray(estimates)
    scale = np.std(draws, axis=0, ddof=1)
    if np.any(scale <= 0):
        raise ValueError('Simultaneous bands require nonconstant contrasts')
    maxima = np.max(np.abs((draws - estimates) / scale), axis=1)
    critical = float(np.quantile(maxima, .95))
    return {'critical': critical, 'standard_errors': scale.tolist(),
            'ci95': np.stack([estimates - critical * scale,
                              estimates + critical * scale], -1).tolist(),
            'empirical_centered_coverage': float(np.mean(maxima <= critical))}


def membership(peak, points, half):
    valid = ((points >= 0) & (points <= 224)).all(-1)
    return ((np.abs(peak[..., None, :] - points) <= half).all(-1) & valid).any(-1)


def primitive_scores(peaks, points, donors, half):
    own = membership(peaks, points[None], half).astype(float)
    other = np.zeros_like(own)
    # Chunking bounds temporary coordinate arrays without changing donors.
    for start in range(0, len(donors), 10):
        schedule = donors[start:start + 10]
        other += membership(peaks[:, schedule], points[None, None], half).sum(1)
    other /= len(donors)
    return own, other


def run(input_dir, manifest_path, output_dir):
    paths = [input_dir / 'new_CUB_context.npz', input_dir / 'new_CUB_native_LocCBM.npz', manifest_path]
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    protocol = {'scope': 'Exploratory after all four point estimates were known; no preregistration or independent confirmation.',
        'fixed': '800 photos, 8323 queries, 40 inherited attributes, own-box/bilinear/row-major, five Gaussian heads, 200 donors, 3000 common image/head draws.',
        'widths': list(WIDTHS), 'width_pairs': [list(v) for v in itertools.combinations(WIDTHS, 2)],
        'multiplicity': 'Separate fixed families: four width values and all six wider-minus-narrower differences; centered-bootstrap maximum absolute standardized deviation, 95% quantile.',
        'anatomical_groups': 'Exact official keypoint sets from attribute prefixes, determined without outcome grouping. Attribute-macro means within every group and every leave-one-group-out subset; all reported.',
        'group_intervals': 'Pointwise joint image/head intervals for description, not simultaneous per-part hypothesis tests.',
        'inputs_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / 'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding='utf-8')) == protocol
    else:
        write_json(protocol_path, protocol)
    with np.load(paths[0], allow_pickle=False) as archive:
        context = {k: archive[k] for k in archive.files}
    with np.load(paths[1], allow_pickle=False) as archive:
        bank = {k: archive[k] for k in archive.files}
    image, group = context['image'], context['group']
    n_images = int(image.max()) + 1
    n_groups = int(group.max()) + 1
    assert len(manifest['attributes']) == n_groups == 40
    incidence = np.zeros((n_images, n_groups))
    np.add.at(incidence, (image, group), 1)
    counts = incidence.sum(0)
    weights = context['weights'].astype(float)
    denominator = weights @ incidence
    assert (denominator > 0).all()
    indices = bank['head_indices']
    draw_index = np.arange(len(weights))[:, None]
    def aggregate(values):
        matrix = np.zeros((n_images, n_groups))
        np.add.at(matrix, (image, group), values)
        return (weights @ matrix) / denominator
    width_results, all_draws = {}, []
    for half in WIDTHS:
        own, donor = primitive_scores(bank['peaks'], context['points_xy'], context['donors'], half)
        gain = own - donor
        attr_gain = np.array([[v[group == a].mean() for a in range(n_groups)] for v in gain])
        attr_draws = np.stack([aggregate(v) for v in gain], axis=1)
        ref_draws = attr_draws[:, 1:][draw_index, indices].mean(1)
        attr_delta = attr_gain[0] - attr_gain[1:].mean(0)
        delta_draws = (attr_draws[:, 0] - ref_draws).mean(1)
        all_draws.append(delta_draws)
        width_results[str(half)] = {'delta_G': float(attr_delta.mean()),
            'pointwise_ci95': interval(delta_draws),
            'S': float(np.mean([own[0, group == a].mean() for a in range(n_groups)])),
            'G': float(attr_gain[0].mean()), 'G_random': float(attr_gain[1:].mean())}
        if half == 45:
            primary = {'own': own, 'donor': donor, 'gain': gain,
                'attr_gain': attr_gain, 'attr_draws': attr_draws,
                'attr_delta': attr_delta, 'delta_draws': attr_draws[:, 0] - ref_draws}
    draws = np.stack(all_draws, axis=1)
    estimates = np.array([width_results[str(h)]['delta_G'] for h in WIDTHS])
    curve_band = simultaneous(estimates, draws)
    pairs = list(itertools.combinations(range(len(WIDTHS)), 2))
    pair_draws = np.stack([draws[:, b] - draws[:, a] for a, b in pairs], axis=1)
    pair_estimates = np.array([estimates[b] - estimates[a] for a, b in pairs])
    pair_band = simultaneous(pair_estimates, pair_draws)
    pairwise = []
    for at, (a, b) in enumerate(pairs):
        pairwise.append({'wider': WIDTHS[b], 'narrower': WIDTHS[a],
            'delta_difference': float(pair_estimates[at]),
            'pointwise_ci95': interval(pair_draws[:, at]),
            'six_comparison_simultaneous_ci95': pair_band['ci95'][at]})
    anatomical = {}
    for attr in manifest['attributes']:
        anatomical.setdefault(attr['part'], []).append(attr['group_id'])
    part_results = []
    for part, attributes in sorted(anatomical.items()):
        keep = np.array(attributes)
        rest = np.setdiff1d(np.arange(n_groups), keep)
        point = primary['attr_gain'][:, keep].mean(1)
        local = primary['delta_draws'][:, keep].mean(1)
        leave = primary['delta_draws'][:, rest].mean(1)
        part_results.append({'part': part, 'attribute_groups': keep.tolist(),
            'attributes': len(keep), 'queries': int(counts[keep].sum()),
            'S': float(np.mean([primary['own'][0, group == a].mean() for a in keep])),
            'G': float(point[0]), 'G_random': float(point[1:].mean()),
            'delta_G': float(primary['attr_delta'][keep].mean()),
            'pointwise_ci95': interval(local),
            'macro_contribution': float(primary['attr_delta'][keep].sum() / n_groups),
            'leave_out_delta_G': float(primary['attr_delta'][rest].mean()),
            'leave_out_pointwise_ci95': interval(leave)})
    attribute_results = []
    for a, metadata in enumerate(manifest['attributes']):
        attribute_results.append({**metadata, 'queries': int(counts[a]),
            'delta_G': float(primary['attr_delta'][a]),
            'pointwise_ci95': interval(primary['delta_draws'][:, a])})
    result = {'protocol': protocol, 'width_results': width_results,
        'four_width_simultaneous_band': curve_band,
        'paired_width_comparisons': pairwise, 'six_pair_band_details': pair_band,
        'parts': part_results, 'attributes': attribute_results,
        'primary_cells': {'trained_own': width_results['45']['S'],
            'trained_donor': float(width_results['45']['S'] - width_results['45']['G']),
            'reference_own': float(np.mean([primary['own'][1:, group == a].mean() for a in range(n_groups)])),
            'reference_donor': float(np.mean([primary['donor'][1:, group == a].mean() for a in range(n_groups)]))}}
    write_json(output_dir / 'results.json', result)
    np.savez_compressed(output_dir / 'bootstrap_draws.npz', width_draws=draws,
                        attribute_delta_draws=primary['delta_draws'])
    print(json.dumps({'paired_widths': pairwise, 'parts': part_results}, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--attribute-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.input_dir, args.attribute_manifest, args.output_dir)


if __name__ == '__main__':
    main()
