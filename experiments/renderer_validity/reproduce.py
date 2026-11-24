"""Recompute the exploratory renderer-response follow-up from frozen inputs."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from renderer_statistics import analyze


def compare(actual, expected, path='root'):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        return max([compare(actual[k], v, path+'.'+k) for k, v in expected.items()] or [0.])
    if isinstance(expected, list):
        assert len(actual) == len(expected), path
        return max([compare(a, b, path+f'[{i}]') for i, (a, b) in enumerate(zip(actual, expected))] or [0.])
    if expected is None or isinstance(expected, (str, bool)):
        assert actual == expected, (path, actual, expected)
        return 0.
    assert np.isfinite(actual), path
    error = abs(actual-expected)
    assert error < 1e-12, (path, actual, expected)
    return error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
        assert actual == expected, f'Input/source fingerprint differs: {name}'
    data = dict(np.load(root/'data/inputs.npz', allow_pickle=False))
    assert data['peaks'].shape == (3, 2, 4, 500, 5, 2)
    assert (data['queries'] != data['alternatives']).all()
    for cohort in ['original200', 'complement300']:
        rows = data[cohort+'_rows']
        donors = data[cohort+'_donors']
        assert (donors != np.arange(len(rows))[None, :, None]).all()
        for key in ['queries', 'alternatives']:
            q = data[key][rows]
            assert np.array_equal(q[donors, np.arange(5)], np.broadcast_to(q, donors.shape))
    assert not np.intersect1d(data['original200_rows'], data['complement300_rows']).size
    assert np.array_equal(np.sort(np.r_[data['original200_rows'], data['complement300_rows']]), np.arange(500))
    assert np.abs(data['logits'][:, :, None]-data['deleted_logits'])[:, data['unchanged']].max() == 0
    results = analyze(data)
    expected = json.loads((root/'data/expected.json').read_text(encoding='utf-8'))
    error = compare(results, expected)
    output = {'status': 'passed', 'input_files_verified': len(manifest['files']),
              'maximum_error': error, 'results': results,
              'scope': 'Frozen-input numerical reproduction. This does not rerender the dataset or retrain any model.'}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f"Renderer-response check: passed ({len(manifest['files'])} files; maximum error {error:.3g})")


if __name__ == '__main__':
    main()
