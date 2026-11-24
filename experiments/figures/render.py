"""Replot Figure 3 from frozen model maps and user-supplied dataset images.

No image synthesis or model inference is performed. Source RGB pixels are read
from CUB and the official FunnyBirds ZIP, then cropped/resized as declared in
data/provenance.json. Numerical inference regeneration is a separate task.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import zipfile

import numpy as np
from PIL import Image


BASE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_images(cub_root, funnybirds_zip, cases):
    images = []
    for case in cases:
        rgb = []
        if case['dataset'] == 'CUB':
            for name, box in zip(case['source_paths'], case['crop_boxes_xyxy']):
                with Image.open(cub_root / 'images' / name) as source:
                    value = source.convert('RGB').crop(tuple(box)).resize(
                        (224, 224), Image.Resampling.BILINEAR)
                    rgb.append(np.asarray(value))
        else:
            with zipfile.ZipFile(funnybirds_zip) as archive:
                for name in case['source_paths']:
                    with Image.open(io.BytesIO(archive.read(name))) as source:
                        rgb.append(np.asarray(source.convert('RGB').resize(
                            (160, 160), Image.Resampling.BILINEAR)))
        for index, value in enumerate(rgb):
            actual = hashlib.sha256(value.tobytes()).hexdigest()
            if actual != case['resized_rgb_sha256'][index]:
                raise ValueError(f"Source pixels differ: {case['source_paths'][index]}")
        images.append(rgb)
    return images


def draw(cases, images, arrays, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import patheffects

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.4,
                         'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 6, figsize=(4.8, 2.42))
    fig.subplots_adjust(left=.005, right=.995, top=.86, bottom=.07,
                        wspace=.035, hspace=.64)
    titles = ['Recipient', 'Donor source', 'Trained / own', 'Trained / donor',
              'Random / own', 'Random / donor']
    for row, (case, rgb, name) in enumerate(zip(cases, images, ['cub', 'fb'])):
        maps = arrays[name + '__maps']
        peaks = arrays[name + '__peaks_xy']
        window = arrays[name + '__window']
        actual_peaks = maps.reshape(4, -1).argmax(1)
        xy = np.stack([actual_peaks % maps.shape[-1],
                       actual_peaks // maps.shape[-1]], -1)
        np.testing.assert_array_equal(peaks, xy)
        hits = window[peaks[:, 1], peaks[:, 0]].astype(int).tolist()
        if hits != case['hits']:
            raise ValueError(f"Hit predicates differ for {name}")
        label = (f"(a) CUB: {case['concept'].replace('::', ': ')}" if row == 0
                 else f"(b) FunnyBirds: {case['part']} variant {case['query']} vs {case['alternative']}")
        fig.text(.005, .974 if row == 0 else .48, label, fontsize=7.0,
                 weight='bold', va='top')
        for col, ax in enumerate(axes[row]):
            ax.imshow(rgb[1 if col == 1 else 0])
            ax.set_axis_off()
            ax.set_title(titles[col], fontsize=6.3, pad=2)
            if col >= 2:
                k = col - 2
                values = maps[k]
                scaled = ((values - values.min()) /
                          max(float(np.ptp(values)), 1e-12))
                ax.imshow(scaled, cmap='magma', vmin=0, vmax=1,
                          alpha=.65 * scaled)
                ax.contour(window, levels=[.5], colors=['#00d4e8'],
                           linewidths=.65)
                point = ax.plot(peaks[k, 0], peaks[k, 1], marker='x',
                                color='white', markersize=4.3,
                                markeredgewidth=.9)[0]
                point.set_path_effects([patheffects.withStroke(
                    linewidth=1.9, foreground='black')])
                ax.text(.025, .025, f"hit={hits[k]}", transform=ax.transAxes,
                        color='white', fontsize=6.0,
                        bbox={'facecolor': 'black', 'alpha': .75,
                              'edgecolor': 'none', 'pad': 1})
    fig.text(.005, .012,
             'Cyan: acceptance boundary   White x: scored peak   Donor maps retain source coordinates',
             fontsize=6.0)
    fig.savefig(out_dir / 'four_cell_examples.pdf')
    fig.savefig(out_dir / 'four_cell_examples.png', dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cub-root', type=Path, required=True,
                        help='CUB_200_2011 directory containing images/')
    parser.add_argument('--funnybirds-zip', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(args.out_dir / '.mplconfig')
    manifest = json.loads((BASE / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        if sha256(BASE / name) != expected:
            raise ValueError(f'Input checksum mismatch: {name}')
    provenance = json.loads((BASE / 'data/provenance.json').read_text(encoding='utf-8'))
    with np.load(BASE / 'data/maps.npz') as archive:
        arrays = {key: archive[key] for key in archive.files}
    images = load_images(args.cub_root, args.funnybirds_zip, provenance['cases'])
    draw(provenance['cases'], images, arrays, args.out_dir)
    result = {'status': 'passed', 'source_images_verified': 4,
              'scored_peaks_verified': 8,
              'scope': 'Frozen-map replot, not model training or inference regeneration.'}
    (args.out_dir / 'verification.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
