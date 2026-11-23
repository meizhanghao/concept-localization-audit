"""Score user-supplied peaks without a frozen expected-results file."""
import argparse
import json
from pathlib import Path

import numpy as np

from .io import load_arrays, read_json, write_json
from .metrics import keypoint_hit


def _integer(name, value, shape=None):
    value = np.asarray(value)
    if not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f'{name} must contain integers')
    if shape is not None and value.shape != shape:
        raise ValueError(f'Invalid {name} shape: {value.shape}; expected {shape}')
    return value.astype(np.int64, copy=False)


def _rectangle_union_area(points, half, side):
    rectangles = []
    for x, y in points:
        if x < 0 or y < 0 or x > side or y > side:
            continue
        left, right = max(0, int(x) - half), min(side - 1, int(x) + half)
        top, bottom = max(0, int(y) - half), min(side - 1, int(y) + half)
        if left <= right and top <= bottom:
            rectangles.append((left, right + 1, top, bottom + 1))
    boundaries = sorted({v for r in rectangles for v in r[:2]})
    area = 0
    for left, right in zip(boundaries, boundaries[1:]):
        intervals = sorted((top, bottom) for x1, x2, top, bottom in rectangles if x1 <= left < x2)
        length = 0; end = -1
        for top, bottom in intervals:
            length += max(0, bottom - max(top, end)); end = max(end, bottom)
        area += (right - left) * length
    return area


def evaluate(arrays, config):
    """Return a paired crossed audit; all eligibility is declared by the caller.

    Peak coordinates are (x,y); the first head is trained, others are references.
    Donors index query records. Image IDs define resampling clusters, group IDs
    define equal-weight macro groups, and concept IDs constrain legal donors.
    """
    peak = _integer('peaks_xy', arrays['peaks_xy'])
    if peak.ndim != 3 or peak.shape[0] < 1 or peak.shape[2] != 2 or peak.shape[1] < 2:
        raise ValueError('peaks_xy must have shape (1 + reference_heads, queries >= 2, 2)')
    heads, queries, _ = peak.shape
    images_raw = np.asarray(arrays['image_id']); group_raw = np.asarray(arrays['group_id'])
    concept = np.asarray(arrays['concept_id'])
    if any(v.shape != (queries,) for v in (images_raw, group_raw, concept)):
        raise ValueError('image_id, group_id and concept_id must be one-dimensional query-length arrays')
    if any(v.dtype.kind not in 'iuUS' for v in (images_raw, group_raw, concept)):
        raise ValueError('IDs must be integer or string arrays (no objects or floating-point IDs)')
    image_names, images = np.unique(images_raw, return_inverse=True)
    group_names, groups = np.unique(group_raw, return_inverse=True)
    if len({(str(i), str(c)) for i, c in zip(images_raw, concept)}) != queries:
        raise ValueError('Duplicate image/concept query records')
    n_images, n_groups = len(image_names), len(group_names)
    if n_images < 2:
        raise ValueError('At least two image clusters are required')
    donors = _integer('donors', arrays['donors'])
    if donors.ndim != 2 or donors.shape[0] < 1 or donors.shape[1] != queries:
        raise ValueError('donors must have shape (schedules >= 1, queries)')
    if (donors < 0).any() or (donors >= queries).any():
        raise ValueError('Donor index out of range')
    if (images[donors] == images[None]).any():
        raise ValueError('Every donor must be from a different image')
    if not (concept[donors] == concept[None]).all():
        raise ValueError('Every donor must preserve the declared concept identity')

    geometry = config['geometry']
    if not isinstance(geometry['side'], int) or isinstance(geometry['side'], bool):
        raise ValueError('geometry.side must be a positive integer')
    side = geometry['side']
    if side < 1 or (peak < 0).any() or (peak >= side).any():
        raise ValueError('Peak coordinates must lie inside the declared square image grid')
    if geometry['type'] == 'keypoints':
        half = geometry['half_width']
        if not isinstance(half, int) or half < 0:
            raise ValueError('half_width must be a nonnegative integer')
        points = _integer('points_xy', arrays['points_xy'])
        if points.ndim != 3 or points.shape[0] != queries or points.shape[2] != 2:
            raise ValueError('points_xy must have shape (queries, keypoints, 2)')
        own = keypoint_hit(peak, points[None], half, side)
        other = keypoint_hit(peak[:, donors], points[None, None], half, side).mean(1)
        chance_per_query = np.array([_rectangle_union_area(p, half, side) for p in points]) / side ** 2
    elif geometry['type'] == 'mask':
        windows = np.asarray(arrays['windows'])
        if windows.shape != (queries, side, side) or not np.isin(windows, [0, 1]).all():
            raise ValueError('windows must be binary and have shape (queries, side, side)')
        own = windows[np.arange(queries)[None], peak[..., 1], peak[..., 0]]
        transferred = peak[:, donors]
        other = windows[np.arange(queries)[None, None], transferred[..., 1], transferred[..., 0]].mean(1)
        chance_per_query = windows.mean((1, 2))
    else:
        raise ValueError('geometry.type must be keypoints or mask')

    counts = np.bincount(groups, minlength=n_groups)
    macro_weights = 1 / (n_groups * counts[groups])
    incidence = np.zeros((n_images, n_groups))
    np.add.at(incidence, (images, groups), 1)
    rng = np.random.default_rng(config.get('seed', 0))
    if 'bootstrap_weights' in arrays:
        weights = _integer('bootstrap_weights', arrays['bootstrap_weights'])
        if weights.ndim != 2 or weights.shape[1] != n_images or weights.shape[0] < 2:
            raise ValueError('Invalid bootstrap_weights dimensions')
        if (weights < 0).any() or not (weights.sum(1) == n_images).all():
            raise ValueError('Bootstrap weights must be nonnegative image-cluster multiplicities summing to image count')
        if not ((weights @ incidence) > 0).all():
            raise ValueError('A supplied bootstrap draw loses a declared macro group')
        proposals = len(weights); retention = 1.
    else:
        draws = int(config.get('draws', 3000))
        if draws < 2 or draws > 100000:
            raise ValueError('draws must be between 2 and 100000')
        accepted = []; retained = 0; proposals = 0; maximum = 6 * draws
        while retained < draws and proposals < maximum:
            batch = min(max(64, draws - retained), maximum - proposals)
            candidate = rng.multinomial(n_images, np.full(n_images, 1 / n_images), size=batch)
            good = ((candidate @ incidence) > 0).all(1)
            accepted.append(candidate[good]); retained += int(good.sum()); proposals += batch
        if retained < draws:
            raise ValueError('Insufficient bootstrap coverage: not enough draws retain every group; change the declared design, not the reported result')
        weights = np.concatenate(accepted)[:draws]; retention = retained / proposals
    denominator = weights @ incidence
    def aggregate(value):
        matrix = np.zeros_like(incidence)
        np.add.at(matrix, (images, groups), value)
        return ((weights @ matrix) / denominator).mean(1)
    def interval(value):
        return np.quantile(value, [.025, .975]).tolist()
    gains = own.astype(float) - other
    points = gains @ macro_weights
    boot = np.array([aggregate(g) for g in gains]).T
    spatial = own @ macro_weights
    result = {'S': float(spatial[0]), 'G': float(points[0]), 'G_ci95': interval(boot[:, 0]),
              'chance': float(chance_per_query @ macro_weights),
              'cells': {'trained_own': float(spatial[0]), 'trained_donor': float(other[0] @ macro_weights)},
              'coverage': {'images': n_images, 'queries': queries, 'macro_groups': n_groups,
                           'group_counts': dict(zip(map(str, group_names), counts.tolist())),
                           'bootstrap_draws': len(weights), 'bootstrap_proposals': proposals,
                           'bootstrap_retained_fraction': float(retention), 'reference_heads': heads - 1},
              'protocol': config,
              'scope': 'Spatial correspondence conditional on supplied maps, eligibility and donors; no semantic or causal certification.'}
    positive_g = result['G_ci95'][0] > 0
    if heads > 1:
        if 'head_indices' in arrays:
            indices = _integer('head_indices', arrays['head_indices'], (len(weights), heads - 1))
            if (indices < 0).any() or (indices >= heads - 1).any():
                raise ValueError('head_indices out of range')
        else:
            indices = rng.integers(0, heads - 1, size=(len(weights), heads - 1))
        reference = boot[:, 1:][np.arange(len(weights))[:, None], indices].mean(1)
        joint = boot[:, 0] - reference
        result.update(G_random=float(points[1:].mean()), delta_G=float(points[0] - points[1:].mean()),
                      image_head_ci95=interval(joint), fixed_head_image_ci95=interval(boot[:, 0] - boot[:, 1:].mean(1)),
                      individual_reference_G=points[1:].tolist())
        result['cells'].update(reference_own=float(spatial[1:].mean()), reference_donor=float((other[1:] @ macro_weights).mean()))
        result['decision'] = ('excess correspondence resolved' if positive_g and result['image_head_ci95'][0] > 0
                              else 'positive G; excess unresolved' if positive_g else 'correspondence unresolved')
    else:
        result['decision'] = 'matched reference unavailable'
    threshold = config.get('s_adequacy_threshold')
    if threshold is not None and (not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1):
        raise ValueError('s_adequacy_threshold must be null or a number in [0,1]')
    result['spatial_requirement'] = ('not specified' if threshold is None else 'met' if result['S'] >= threshold else 'not met')
    return result


def markdown_report(result):
    lines = ['# Crossed localization audit', '', f"Decision: **{result['decision']}**.",
             f"Spatial requirement: {result['spatial_requirement']}.", '', '| Quantity | Value |', '|---|---:|']
    for key in ('S', 'G', 'G_random', 'delta_G', 'chance'):
        if key in result:
            lines.append(f'| {key} | {result[key]:.6f} |')
    for key in ('G_ci95', 'image_head_ci95', 'fixed_head_image_ci95'):
        if key in result:
            low, high = result[key]; lines.append(f'| {key} | [{low:.6f}, {high:.6f}] |')
    lines += ['', '## Coverage', '', '```json', json.dumps(result['coverage'], indent=2), '```',
              '', '## Declared protocol', '', '```json', json.dumps(result['protocol'], indent=2), '```',
              '', result['scope'],
              'Intervals condition on fixed donors and the trained model. Reference heads are resampled from the supplied finite bank.',
              'An interval crossing zero is unresolved, not an equivalence result.', '']
    return '\n'.join(lines)


def main(argv=None):
    command = argparse.ArgumentParser(description='Audit new peak inputs without expected results.')
    command.add_argument('--input', type=Path, required=True)
    command.add_argument('--config', type=Path, required=True)
    command.add_argument('--output-dir', type=Path, required=True)
    arguments = command.parse_args(argv)
    try:
        result = evaluate(load_arrays(arguments.input), read_json(arguments.config))
        output = arguments.output_dir.resolve()
        if output in [arguments.input.resolve(), arguments.config.resolve()]:
            raise ValueError('Output directory must differ from the input files')
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / 'audit.json', result)
        (output / 'audit.md').write_text(markdown_report(result), encoding='utf-8')
        print(f"{result['decision']}; S={result['S']:.6f}, G={result['G']:.6f}")
        print(f'Report saved to {output}')
        return 0
    except (OSError, ValueError, KeyError) as error:
        command.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
