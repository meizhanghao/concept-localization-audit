"""Command-line entry point for the frozen numerical audits."""
import argparse
import csv
import os
from pathlib import Path
import platform
import sys
import time

from . import __version__

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parser():
    command = argparse.ArgumentParser(description='Reproduce concept-localization audit scores from frozen numerical inputs.')
    command.add_argument('--audit', choices=('all', 'core', 'concepts', 'bridge', 'extensions'), default='all',
                         help='Analysis to run (default: all).')
    command.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data',
                         help='Directory containing checksums.json and the audit input folders.')
    command.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'results',
                         help='Result directory (default: results/ in the project).')
    command.add_argument('--verify-only', action='store_true', help='Verify input fingerprints without running scores.')
    command.add_argument('--threads', type=int, default=2, help='BLAS thread count (default: 2).')
    command.add_argument('--version', action='version', version=__version__)
    return command


def write_core_csv(path, cases):
    columns = ('case', 'S', 'G', 'G_random', 'delta_G', 'image_head_ci95',
               'fixed_head_image_ci95', 'gain_difference', 'paired_image_ci95')
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, values in cases.items():
            writer.writerow({'case': name, **{key: values[key] for key in columns[1:] if key in values}})


def main(argv=None):
    command = parser()
    arguments = command.parse_args(argv)
    if arguments.threads < 1:
        command.error('--threads must be positive')
    # Set limits before importing NumPy or any numerical audit module.
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = str(arguments.threads)
    import numpy as np
    from . import bridge, concepts, core, extensions
    from .io import verify_inputs, write_json

    started = time.perf_counter()
    try:
        data_root = arguments.data_dir.resolve()
        count = verify_inputs(data_root)
        print(f'Inputs verified: {count} files', flush=True)
        if arguments.verify_only:
            return 0
        output = arguments.output_dir.resolve()
        if output == data_root or data_root in output.parents:
            raise ValueError('Choose an output directory outside data/ to preserve frozen inputs')
        output.mkdir(parents=True, exist_ok=True)
        functions = {'core': core.run, 'concepts': concepts.run, 'bridge': bridge.run, 'extensions': extensions.run}
        selected = tuple(functions) if arguments.audit == 'all' else (arguments.audit,)
        summary = {'status': 'passed', 'version': __version__, 'python': platform.python_version(),
                   'numpy': np.__version__, 'input_files_verified': count, 'checks': {}}
        for name in selected:
            result = functions[name](data_root)
            filename = name + '.json'
            write_json(output / filename, result)
            summary['checks'][name] = {'status': result['status'],
                                      'maximum_error': result['maximum_error'], 'result': filename}
            if name == 'core':
                write_core_csv(output / 'core_scores.csv', result['cases'])
            print(f"{name}: passed (maximum error {result['maximum_error']:.3g})", flush=True)
        summary['runtime_seconds'] = time.perf_counter() - started
        write_json(output / 'verification.json', summary)
        print(f'Results saved to {output}', flush=True)
        return 0
    except (OSError, ValueError, KeyError, AssertionError) as error:
        print(f'Audit failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
