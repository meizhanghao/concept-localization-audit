"""Rescore the released native CUB cache at four declared window widths.

This is an exploratory follow-up on the existing 40-attribute cohort. It does
not select a width, change the original h=45 operating point, or train a model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from audit.io import load_arrays
from audit.new_input import evaluate

WIDTHS = (16, 24, 36, 45)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results/geometry'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data = root / 'data/core_audit/data'
    paths = [data / 'new_CUB_context.npz', data / 'new_CUB_native_LocCBM.npz']
    protocol = {
        'scope': 'Exploratory follow-up after h=45 outcomes; no confirmatory or multiplicity-adjusted claim.',
        'half_widths': list(WIDTHS), 'side': 224,
        'fixed': '800 recipient photos, 40 inherited attributes, 8323 queries, native own-box/bilinear/row-major peaks, five Gaussian heads, 200 donors, 3000 paired image/head resamples.',
        'selection': 'Report every width; h=45 remains primary. No new peak, query, head, donor or adapter selection.',
        'inputs': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    target = args.output_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    protocol_path = target / 'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding='utf-8')) == protocol
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    context, bank = map(load_arrays, paths)
    arrays = {'peaks_xy': bank['peaks'], 'image_id': context['image'],
        'group_id': context['group'], 'concept_id': context['group'],
        'points_xy': context['points_xy'], 'donors': context['donors'],
        'bootstrap_weights': context['weights'], 'head_indices': bank['head_indices']}
    start = time.perf_counter()
    results = {}
    for half in WIDTHS:
        config = {'geometry': {'type': 'keypoints', 'side': 224, 'half_width': half},
                  'eligibility': 'Fixed 40 inherited attributes; all 8323 eligible queries on 800 disjoint photos.'}
        results[str(half)] = evaluate(arrays, config)
        value = results[str(half)]
        print(f"h={half}: S={value['S']:.6f}, G={value['G']:.6f}, "
              f"G_rand={value['G_random']:.6f}, delta_G={value['delta_G']:.6f}, "
              f"CI={value['image_head_ci95']}", flush=True)
    output = {'protocol': protocol, 'results': results,
              'elapsed_seconds': time.perf_counter() - start}
    (target / 'results.json').write_text(json.dumps(output, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
