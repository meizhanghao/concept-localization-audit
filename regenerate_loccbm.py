"""Regenerate a small native pre-mask Loc-CBM example from user-owned CUB images.

This script uses torchvision primitives and the author-released checkpoint. It
does not reproduce training, masked concept scores, or species classification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cub-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).parent / 'examples/loccbm/manifest.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in ['TORCH_HOME', 'CUDA_CACHE_PATH', 'TMP', 'TEMP']:
        folder = args.output_dir / 'runtime_cache' / key.lower()
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder.resolve())
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from torch.nn import functional as F
    from torchvision import transforms
    from torchvision.models import resnet50
    from PIL import Image

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    checksum = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if checksum != manifest['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA256 differs from the released checkpoint specified in the manifest')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    backbone = resnet50(weights=None)
    backbone.avgpool = torch.nn.Identity(); backbone.fc = torch.nn.Identity()
    backbone.load_state_dict({k.removeprefix('backbone.'): v for k, v in state.items() if k.startswith('backbone.')}, strict=True)
    backbone.to(device).eval()
    transform = transforms.Compose([transforms.Resize(512), transforms.CenterCrop(448), transforms.ToTensor(),
                                    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    source = args.manifest.parent
    with np.load(source / 'reference_rows.npz', allow_pickle=False) as rows:
        reference = rows['reference_rows'].copy()
    attribute = manifest['attribute_id_0based']
    weights = torch.cat([state['classifier.concept_proj.weight'][attribute][None], torch.from_numpy(reference)]).to(device)
    images = []; grids = []
    yy, xx = np.mgrid[:56, :56]; xx = (xx + .5) / 56; yy = (yy + .5) / 56
    root = args.cub_root.resolve()
    for record in manifest['records']:
        path = (root / 'images' / record['image_path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Image path escapes the supplied dataset directory')
        with Image.open(path) as image:
            image = image.convert('RGB'); width, height = image.size
            images.append(transform(image))
        rw, rh = (512, int(512 * height / width)) if width <= height else (int(512 * width / height), 512)
        left, top = round((rw - 448) / 2), round((rh - 448) / 2)
        x1, y1, x2, y2 = record['box_xyxy']
        gx = 2 * (((x1 + xx * (x2 - x1)) / width * rw - left) / 448) - 1
        gy = 2 * (((y1 + yy * (y2 - y1)) / height * rh - top) / 448) - 1
        grids.append(np.stack([gx, gy], -1))
    x = torch.stack(images).to(device)
    with torch.no_grad():
        f = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(x))))
        for layer in [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]:
            f = layer(f)
        # Each row is the same native concept projection or one fixed reference.
        native = torch.einsum('bdhw,cd->bchw', f, weights).relu()
        projected = F.grid_sample(native, torch.as_tensor(np.array(grids), dtype=torch.float32, device=device),
                                  mode='bilinear', padding_mode='zeros', align_corners=False)
        zero = projected.sum((-2, -1)) <= 1e-12
        projected = torch.where(zero[..., None, None], torch.ones_like(projected), projected)
        large = F.interpolate(projected, (224, 224), mode='bicubic', align_corners=False)
        index = large.flatten(2).argmax(-1).cpu().numpy()
        peaks = np.stack([index % 224, index // 224], -1).transpose(1, 0, 2)
    q = len(manifest['records'])
    donors = np.array([np.roll(np.arange(q), 1)])
    np.savez_compressed(args.output_dir / 'peaks.npz', peaks_xy=peaks,
                        image_id=np.array([r['image_id'] for r in manifest['records']]),
                        group_id=np.zeros(q, dtype=int), concept_id=np.full(q, attribute),
                        points_xy=np.array([r['points_xy'] for r in manifest['records']]), donors=donors)
    np.savez_compressed(args.output_dir / 'native_maps.npz', native_maps=native.cpu().numpy(), zero=zero.cpu().numpy())
    config = {'geometry': {'type': 'keypoints', 'side': 224, 'half_width': 45}, 'seed': 2026112308,
              'draws': 3000, 's_adequacy_threshold': None,
              'eligibility': 'Fixed first eight eligible records of one declared attribute; not the full paper cohort.',
              'donors': 'One cyclic same-attribute donor schedule within these eight records.',
              'reference': 'Five saved Gaussian rows from the original native reference bank; no regenerated RNG claim.'}
    (args.output_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    with np.load(source / 'verification.npz', allow_pickle=False) as expected:
        mismatches = int((peaks != expected['peaks_xy']).sum())
        error = float(np.abs(native.cpu().numpy().transpose(1, 0, 2, 3) - expected['native_maps']).max())
    report = {'checkpoint_sha256': checksum, 'device': str(device), 'torch': torch.__version__,
              'images': q, 'heads': len(weights), 'peak_coordinate_mismatches': mismatches,
              'native_map_max_absolute_error': error, 'exact_saved_peak_match': mismatches == 0,
              'scope': 'Small raw-image pre-mask map example, not full training or full-cohort regeneration.',
              'portability': 'CPU/GPU/library differences can move near-tied bicubic maxima; report mismatches rather than silently replace them.'}
    (args.output_dir / 'regeneration.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
