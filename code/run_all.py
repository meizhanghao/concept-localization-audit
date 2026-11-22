"""Run the frozen portable checks, with no imports from the source project."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parent
    large = json.loads((root / 'large_inputs.json').read_text(encoding='utf-8'))
    for record in large:
        destination = root / record['path']
        if not destination.exists():
            parts = []
            for chunk in record['chunks']:
                raw = (root / chunk['path']).read_bytes()
                assert hashlib.sha256(raw).hexdigest() == chunk['sha256'], chunk['path']
                parts.append(raw)
            joined = b''.join(parts)
            assert len(joined) == record['bytes']
            assert hashlib.sha256(joined).hexdigest() == record['sha256']
            destination.parent.mkdir(exist_ok=True, parents=True)
            destination.write_bytes(joined)
    manifest = json.loads((root / 'release_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    output = root / '.results'
    output.mkdir(exist_ok=True)
    tasks = [
        ('core_audit', ['--output', str(output / 'core_audit.json')], output / 'core_audit.json'),
        ('concept_eligibility', [], root / 'concept_eligibility/results.json'),
        ('donor_bridge', [], root / 'donor_bridge/results.json'),
    ]
    summary = {'status': 'passed', 'python': platform.python_version(), 'checks': {}}
    started = time.perf_counter()
    for name, arguments, result in tasks:
        process = subprocess.run([sys.executable, '-I', '-B', str(root / name / 'rescore.py'), *arguments],
                                 cwd=root, capture_output=True, text=True,
                                 env={**os.environ, 'OPENBLAS_NUM_THREADS': '2',
                                      'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
        (output / (name + '.log')).write_text(process.stdout + process.stderr, encoding='utf-8')
        if process.returncode:
            raise RuntimeError(f'{name} failed; see .results/{name}.log')
        record = json.loads(result.read_text(encoding='utf-8'))
        assert record['status'] == 'passed', name
        summary['checks'][name] = {
            'status': record['status'], 'result': str(result.relative_to(root)),
            'sha256': hashlib.sha256(result.read_bytes()).hexdigest()}
        print(name + ': passed', flush=True)
    summary['runtime_seconds'] = time.perf_counter() - started
    (output / 'verification.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
