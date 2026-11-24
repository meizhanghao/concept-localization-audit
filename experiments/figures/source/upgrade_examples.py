"""Traceable real-image examples for the existing four audit cells."""
from .common import DATA, OUT, configure, write_json
from .models import ImageConceptModel, load_meta
from .cub_frozen_replication import geometry
from .adapter_sensitivity import grid
from .published_metric_gate import peaks224
from .frozen_adapter_review import hit
from .revision_stage1 import digest
from .controlled_localization import fields, normalize
from pathlib import Path
import json
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

DEST = OUT / 'upgrade_examples_2026-11-23'
FROZEN = OUT / 'cub_frozen_replication_2026-11-20'
RUN = OUT / 'controlled_audit_2026-11-19'
FB = OUT / 'funnybirds_2026-11-15'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare():
    DEST.mkdir(exist_ok=True)
    cub = np.load(FROZEN / 'population.npz'); fb = np.load(RUN / 'stage2_population.npz')
    masks = np.load(FB / 'test_masks.npy', mmap_mode='r')
    valid = [j for j in range(5) if masks[fb['ids'][0], j].any()]
    j = valid[0]
    donor_fb = int(np.load(RUN / 'stage2_query_results.npz')['donors_same_contrast'][0, 0, j])
    paths = [FROZEN / 'population.npz', FROZEN / 'native_peaks.npz',
             OUT / 'frozen_adapter_review_2026-11-21/native_maps_before_projection.npz',
             RUN / 'stage2_population.npz', RUN / 'stage2_query_results.npz',
             RUN / 'stage4_random_peaks.npz', RUN / 'peaks_resnet18_seed0.npz', FB / 'model.pt']
    protocol = {'scope': 'Illustrations only, no population inference or selection by maps, scores, correctness or confidence.',
                'selection': 'CUB first stored eligible query; FB first audit image and first nonempty part in declared part order. Use donor schedule0 and original Gaussian reference head0.',
                'cub_query_index': 0, 'cub_donor_query_index': int(cub['donors'][0, 0]),
                'fb_image_index': 0, 'fb_part': j, 'fb_donor_image_index': donor_fb,
                'display': 'Own-bird-box crops on CUB and fixed160 RGB on FB. Every transferred map is displayed on the recipient frame without keypoint alignment. Per-map display scaling never changes peaks or hit scores.',
                'reference': 'One individual Gaussian head is shown; population G_random averages scores across heads, not heatmaps.',
                'inputs': {str(p): digest(p) for p in paths}}
    path = DEST / 'protocol.json'
    if path.exists():
        assert read(path) == protocol
    else:
        write_json(path, protocol)
    return protocol


def cub_case(protocol):
    pop = np.load(FROZEN / 'population.npz'); meta = load_meta()
    ctx = geometry(meta, pop['rows']); g = grid(ctx, 'own_box')
    raw = dict(np.load(OUT / 'frozen_adapter_review_2026-11-21/native_maps_before_projection.npz'))
    cached = np.load(FROZEN / 'native_peaks.npz')
    ix = [protocol['cub_query_index'], protocol['cub_donor_query_index']]
    i, d = pop['image'][ix]; si, sd = pop['slot'][ix]
    assert pop['attribute'][ix[0]] == pop['attribute'][ix[1]]
    maps = torch.as_tensor(np.stack([raw['head0'][i, si], raw['head0'][d, sd],
                                     raw['head1'][i, si], raw['head1'][d, sd]])[:, None], device='cuda')
    projected = F.grid_sample(maps, g[[i, d, i, d]], mode='bilinear', padding_mode='zeros', align_corners=False)
    zero = projected.sum((-2, -1)) <= 1e-12
    projected = torch.where(zero[..., None, None], torch.ones_like(projected), projected)
    peak = peaks224(projected.cpu().numpy(), device='cuda')[:, 0]
    expected = np.stack([cached['head0'][i, si], cached['head0'][d, sd],
                         cached['head1'][i, si], cached['head1'][d, sd]])
    assert np.array_equal(peak, expected)
    large = F.interpolate(projected, size=(224, 224), mode='bicubic', align_corners=False)[:, 0].cpu().numpy()
    rgb = []
    for image_index in [i, d]:
        row = pop['rows'][image_index]; ident = int(pop['image_ids'][image_index])
        with Image.open(DATA / 'images' / str(meta['paths'][row])) as im:
            rgb.append(np.array(im.convert('RGB').crop(ctx['bbox'][ident]).resize((224, 224), Image.Resampling.BILINEAR)))
    points = pop['points_xy'][ix[0]]
    yy, xx = np.mgrid[:224, :224]
    window = hit(np.stack([xx, yy], -1), points)
    hits = hit(peak, points).astype(int)
    attribute_id = int(pop['original_attribute_ids'][pop['attribute'][ix[0]]])
    path = DATA / 'attributes.txt'
    if not path.exists():
        path = DATA / 'attributes/attributes.txt'
    names = [line.split(maxsplit=1)[1] for line in path.read_text().splitlines()]
    info = {'dataset': 'CUB', 'recipient_image_id': int(pop['image_ids'][i]),
            'donor_image_id': int(pop['image_ids'][d]), 'attribute_id_0based': attribute_id,
            'concept': names[attribute_id], 'reference_seed': 2026111921,
            'peaks_xy': peak.tolist(), 'hits': hits.tolist(), 'zero_maps': zero[:, 0].cpu().tolist(),
            'source_paths': [str(meta['paths'][pop['rows'][x]]) for x in [i, d]],
            'cached_peaks_match': True}
    return rgb, large, peak, window, info


def fb_case(protocol):
    torch.backends.cudnn.allow_tf32 = True
    pop = np.load(RUN / 'stage2_population.npz'); old = np.load(RUN / 'stage4_random_peaks.npz')
    train_peaks = np.load(RUN / 'peaks_resnet18_seed0.npz')['gradcam']
    rgb_all = np.load(FB / 'test_rgb.npy', mmap_mode='r'); masks = np.load(FB / 'test_masks.npy', mmap_mode='r')
    saved = torch.load(FB / 'model.pt', map_location='cpu', weights_only=False)
    model = ImageConceptModel(count=31).cuda().eval(); model.load_state_dict(saved['state_dict'])
    j = protocol['fb_part']; ids = [protocol['fb_image_index'], protocol['fb_donor_image_index']]
    own_maps = []; random_maps = []
    for image_index in ids:
        # Match the original eight-image inference batches to avoid shape-related rounding differences.
        start = (image_index // 8) * 8; batch_ids = pop['ids'][start:start + 8]
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(rgb_all[batch_ids]), device='cuda').float() / 255
            spatial = model(x, True)[2]
            q = torch.as_tensor(pop['queries'][start:start + 8], device='cuda')
            a = torch.as_tensor(pop['alternatives'][start:start + 8], device='cuda')
            for w, output in [(model.head.weight, own_maps), (torch.as_tensor(old['weights'][0], device='cuda'), random_maps)]:
                v = F.interpolate(torch.einsum('bqc,bchw->bqhw', w[q] - w[a], spatial).relu(),
                                  size=(160, 160), mode='bilinear', align_corners=False)
                output.append(normalize(v)[image_index - start, j].cpu().numpy())
    maps = np.array([own_maps[0], own_maps[1], random_maps[0], random_maps[1]])
    index = maps.reshape(4, -1).argmax(1); peak = np.stack([index % 160, index // 160], -1)
    expected_yx = np.array([train_peaks[ids[0], j], train_peaks[ids[1], j],
                            old['head0__gradcam'][ids[0], j], old['head0__gradcam'][ids[1], j]])
    assert np.array_equal(peak[:, ::-1], expected_yx)
    recipient_mask = masks[pop['ids'][ids[0]], j][None, None].astype(bool)
    distance, _ = fields(recipient_mask); window = distance[0, 0, 0] <= 24
    hits = window[peak[:, 1], peak[:, 0]].astype(int)
    meta = np.load(FB / 'metadata.npz'); parts = ['beak', 'eye', 'foot', 'tail', 'wing']
    rgb = [np.asarray(rgb_all[pop['ids'][x]]).transpose(1, 2, 0) for x in ids]
    info = {'dataset': 'FunnyBirds', 'recipient_test_index': int(pop['ids'][ids[0]]),
            'donor_test_index': int(pop['ids'][ids[1]]), 'part': parts[j],
            'query': int(pop['queries'][ids[0], j]), 'alternative': int(pop['alternatives'][ids[0], j]),
            'reference_generator_seed': 821, 'reference_head': 0, 'peaks_xy': peak.tolist(),
            'hits': hits.tolist(), 'cached_peaks_match': True,
            'source_paths': [f"FunnyBirds/test/{meta['test_classes'][pop['ids'][x]]}/{pop['ids'][x]:06d}.png" for x in ids]}
    return rgb, maps, peak, window, info


def render(cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.4, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 6, figsize=(4.8, 2.42))
    fig.subplots_adjust(left=.005, right=.995, top=.86, bottom=.07, wspace=.035, hspace=.64)
    titles = ['Recipient', 'Donor source', 'Trained / own', 'Trained / donor', 'Random / own', 'Random / donor']
    for row, (rgb, maps, peaks, window, info) in enumerate(cases):
        label = (f"(a) CUB: {info['concept'].replace('::', ': ')}" if row == 0
                 else f"(b) FunnyBirds: {info['part']} variant {info['query']} vs {info['alternative']}")
        fig.text(.005, .974 if row == 0 else .48, label, fontsize=7.0, weight='bold', va='top')
        for col, ax in enumerate(axes[row]):
            ax.imshow(rgb[1 if col == 1 else 0]); ax.set_axis_off()
            ax.set_title(titles[col], fontsize=6.3, pad=2)
            if col >= 2:
                k = col - 2; values = maps[k]
                scaled = (values - values.min()) / max(float(np.ptp(values)), 1e-12)
                ax.imshow(scaled, cmap='magma', vmin=0, vmax=1, alpha=.65 * scaled)
                ax.contour(window, levels=[.5], colors=['#00d4e8'], linewidths=.65)
                point = ax.plot(peaks[k, 0], peaks[k, 1], marker='x', color='white', markersize=4.3, markeredgewidth=.9)[0]
                point.set_path_effects([patheffects.withStroke(linewidth=1.9, foreground='black')])
                ax.text(.025, .025, f"hit={info['hits'][k]}", transform=ax.transAxes, color='white', fontsize=6.0,
                        bbox={'facecolor': 'black', 'alpha': .75, 'edgecolor': 'none', 'pad': 1})
    fig.text(.005, .012, 'Cyan: acceptance boundary   White x: scored peak   Donor maps retain source coordinates', fontsize=6.0)
    fig.savefig(DEST / 'four_cell_examples.pdf')
    fig.savefig(DEST / 'four_cell_examples.png', dpi=300)
    plt.close(fig)


def main():
    protocol = prepare(); configure(0)
    cases = [cub_case(protocol), fb_case(protocol)]
    arrays = {}
    for name, case in zip(['cub', 'fb'], cases):
        rgb, maps, peak, window, info = case
        arrays.update({f'{name}__rgb': np.array(rgb), f'{name}__maps': maps,
                       f'{name}__peaks_xy': peak, f'{name}__window': window})
    np.savez_compressed(DEST / 'example_arrays.npz', **arrays)
    write_json(DEST / 'examples.json', {'cases': [case[-1] for case in cases],
               'all_eight_peak_checks_passed': True, 'display_only_individual_reference': True,
               'selection_rule': protocol['selection'], 'source_sha256': digest(Path(__file__))})
    render(cases)
    assert all(digest(Path(p)) == h for p, h in protocol['inputs'].items())
    print(json.dumps({'status': 'passed', 'cases': [case[-1] for case in cases]}), flush=True)


if __name__ == '__main__':
    main()
