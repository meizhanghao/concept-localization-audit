"""Rescore the author-released CEM using the common frozen CUB queries."""
import argparse
import json
from pathlib import Path

from audit.io import load_arrays, verify_inputs
from audit.new_input import evaluate, markdown_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results/released_cem'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    verify_inputs(root / 'data')
    context = load_arrays(root / 'data/core_audit/data/new_CUB_context.npz')
    bank = load_arrays(root / 'data/core_audit/data/new_CUB_released_CEM.npz')
    result = evaluate({'peaks_xy': bank['peaks'], 'points_xy': context['points_xy'],
        'donors': context['donors'], 'image_id': context['image'],
        'concept_id': context['group'], 'group_id': context['group'],
        'bootstrap_weights': context['weights'], 'head_indices': bank['head_indices']},
        {'geometry': {'type': 'keypoints', 'side': 224, 'half_width': 45},
         'eligibility': 'Exactly existing800 photos/8323 queries/40 attributes; no CEM outcome filtering.'})
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (out / 'audit.md').write_text(markdown_report(result), encoding='utf-8')
    print(f"G={result['G']:.6f}, delta_G={result['delta_G']:.6f}, joint CI={result['image_head_ci95']}")


if __name__ == '__main__':
    main()
