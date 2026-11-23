"""Regenerate released-CEM peaks from CUB photos and the author's checkpoint.

Use --limit-images 8 for the small example or 800 for the full frozen cohort.
CUDA is required to reproduce the published reference bank's RNG sequence.
No training or intervention is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from audit.cem_model import CEMInference, parameter_bank, cam_maps, crop_origin, scored_peaks
from audit.io import load_arrays


CHECKPOINT_SHA256 = '386a366087713da40a56c150c09d9f832a5a05faa748c04cb7490a996934d212'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cub-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--limit-images', type=int, default=8)
    parser.add_argument('--output-dir', type=Path, default=Path('results/raw_cem'))
    args = parser.parse_args()
    if not 1 <= args.limit_images <= 800:
        parser.error('--limit-images must be in [1,800]')
    if not torch.cuda.is_available():
        parser.error('This raw reference example requires CUDA; CPU numerical rescoring is audit_released_cem.py')
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        parser.error('Checkpoint differs from the documented author release')
    root = Path(__file__).resolve().parent
    context = load_arrays(root / 'data/core_audit/data/new_CUB_context.npz')
    expected = load_arrays(root / 'data/core_audit/data/new_CUB_released_CEM.npz')
    manifest = json.loads((root / 'data/core_audit/attribute_manifest.json').read_text())
    protocol = json.loads((root / 'docs/released_cem_protocol.json').read_text())
    assert [a['original_attribute_id_0based'] for a in manifest['attributes']] == [protocol['attribute_order'][a] for a in protocol['fixed40_cem_indices']]
    mapping = np.array(protocol['fixed40_cem_indices'])
    ids = np.load(root / 'data/core_audit/cub_frozen_image_ids.npy', allow_pickle=False)[:args.limit_images]
    names = {int(line.split()[0]): line.split()[1] for line in (args.cub_root / 'images.txt').read_text().splitlines()}
    boxes = {}
    for line in (args.cub_root / 'bounding_boxes.txt').read_text().splitlines():
        ident, x, y, width, height = line.split()
        x, y, width, height = [int(float(v)) for v in (x, y, width, height)]
        boxes[int(ident)] = x, y, x + width, y + height
    transform = transforms.Compose([transforms.CenterCrop(299), transforms.ToTensor(),
                                    transforms.Normalize([.5] * 3, [2.] * 3)])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = CEMInference(state).cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bank = parameter_bank(model)
    selected = np.flatnonzero(context['image'] < len(ids))
    peaks = np.empty((6, len(selected), 2), dtype=np.int16)
    yy, xx = np.mgrid[:56, :56]
    with torch.no_grad():
        for start in range(0, len(ids), 16):
            end = min(start + 16, len(ids)); images, grids = [], []
            for ident in ids[start:end]:
                with Image.open(args.cub_root / 'images' / names[int(ident)]) as im:
                    im = im.convert('RGB'); width, height = im.size
                    images.append(transform(im))
                x1, y1, x2, y2 = boxes[int(ident)]
                gx = 2 * (x1 + (xx + .5) * (x2 - x1) / 56 - crop_origin(width)) / 299 - 1
                gy = 2 * (y1 + (yy + .5) * (y2 - y1) / 56 - crop_origin(height)) / 299 - 1
                grids.append(np.stack([gx, gy], -1))
            grid = torch.tensor(np.array(grids), dtype=torch.float32, device='cuda')
            spatial, pre = model.spatial_forward(torch.stack(images).cuda())
            query_indices = np.flatnonzero((context['image'][selected] >= start) & (context['image'][selected] < end))
            q = selected[query_indices]
            for h, params in enumerate(bank):
                maps = cam_maps(spatial, pre, params, model.pre_concept_model.fc.weight, mapping)
                coordinate, _ = scored_peaks(maps, grid)
                peaks[h, query_indices] = coordinate.cpu().numpy()[context['image'][q] - start, context['group'][q]]
    mismatches = int((peaks != expected['peaks'][:, selected]).any(-1).sum())
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'peaks.npz', peaks=peaks, query_indices=selected)
    report = {'images': len(ids), 'queries': len(selected), 'heads': 6,
        'peak_coordinates_checked': int(peaks.shape[0] * peaks.shape[1]),
        'peak_mismatches': mismatches, 'checkpoint_sha256': CHECKPOINT_SHA256,
        'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
        'scope': 'Raw CUDA map example, not independent model training or hardware-invariant tie certification.'}
    (output / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if mismatches:
        raise SystemExit('Peak differences detected; inspect near ties and runtime compatibility before claiming exact reproduction.')


if __name__ == '__main__':
    main()
