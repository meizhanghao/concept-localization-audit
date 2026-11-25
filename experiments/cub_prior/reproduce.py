"""Recompute the CUB position-prior card from frozen derived numerical inputs."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

from cub_prior_statistics import analyze, training_centers


BASE = Path(__file__).resolve().parent
FAMILIES = [[1], [3], [5], [9], [14], [6, 10], [2], [0], [7, 11], [4]]


def compare(observed, expected):
    """Compare computed results only after analysis, without expected inputs."""
    errors = []
    if isinstance(expected, dict):
        for key, value in expected.items():
            errors.extend(compare(observed[key], value))
    elif isinstance(expected, list):
        if len(observed) != len(expected):
            raise ValueError('Result lengths differ')
        for a, b in zip(observed, expected):
            errors.extend(compare(a, b))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not np.isfinite(observed):
            raise ValueError('Nonfinite numerical result')
        errors.append(abs(observed - expected))
    elif observed != expected:
        raise ValueError(f'Result mismatch: {observed!r} != {expected!r}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('results/cub_prior.json'))
    args = parser.parse_args()
    manifest = json.loads((BASE / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        digest = hashlib.sha256((BASE / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f'Checksum mismatch: {name}')
    with np.load(BASE / 'data/inputs.npz') as archive:
        data = {key: archive[key] for key in archive.files}
    centers, counts = training_centers(data['training_points_in_box'],
                                       data['training_visible'], FAMILIES)
    np.testing.assert_allclose(centers, data['training_centers'], atol=1e-6, rtol=0)
    result = analyze(data)
    expected = json.loads((BASE / 'data/expected.json').read_text(encoding='utf-8'))
    errors = compare(result, expected)
    maximum = max(errors, default=0.)
    if maximum > 1e-12:
        raise ValueError(f'Numerical mismatch: {maximum}')
    report = {'status': 'passed', 'maximum_error': maximum,
              'training_centers_recomputed': len(centers),
              'visible_training_points': counts.tolist(), 'results': result,
              'scope': 'Frozen-input prior/scoring reproduction; no model training or raw-image inference.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'maximum_error': maximum,
                      'prior_S': result['prior_S'], 'prior_G': result['prior_G'],
                      'native_minus_prior': result['methods']['native_LocCBM']['model_minus_prior']}, indent=2))


if __name__ == '__main__':
    main()
