"""Input integrity and numerical regression checks shared by the audits."""
import hashlib
import json
from pathlib import Path

import numpy as np

TOLERANCE = 1e-12


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def verify_inputs(data_root):
    """Check input bytes before any score is computed."""
    root = Path(data_root).resolve()
    manifest = read_json(root / 'checksums.json')
    if manifest['algorithm'] != 'sha256':
        raise ValueError('Unsupported input fingerprint algorithm')
    for name, expected in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'Input path escapes data directory: {name}')
        if not path.is_file():
            raise FileNotFoundError(f'Missing input: {name}. Copy the complete data/ directory.')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Input checksum mismatch: {name}')
    return len(manifest['files'])


def check_expected(actual, expected, path='', tolerance=TOLERANCE):
    """Validate reported numerical fields and return the largest absolute error."""
    errors = []
    for key, reference in expected.items():
        field = f'{path}/{key}'
        if isinstance(reference, dict):
            errors.append(check_expected(actual[key], reference, field, tolerance))
        elif isinstance(reference, (list, float, int)):
            value = np.asarray(actual[key])
            target = np.asarray(reference)
            if value.shape != target.shape or not np.isfinite(value).all():
                raise ValueError(f'Invalid result shape or non-finite value: {field}')
            error = float(np.max(np.abs(value - target)))
            if error > tolerance:
                raise ValueError(f'Numerical mismatch at {field}: {error:.3g} > {tolerance:.3g}')
            errors.append(error)
    return max(errors, default=0.0)
