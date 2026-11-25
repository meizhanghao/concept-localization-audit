"""NumPy-only statistics for an image-independent, training-calibrated prior."""
import numpy as np


def training_centers(points_in_box, visible, families):
    """Reuse the historical point-weighted mean; do not weight by test queries."""
    points_in_box = np.asarray(points_in_box)
    visible = np.asarray(visible, dtype=bool)
    if points_in_box.shape != visible.shape + (2,):
        raise ValueError('Expected image x keypoint x xy and matching visibility')
    centers, counts = [], []
    for indices in families:
        values = points_in_box[:, indices][visible[:, indices]]
        if not len(values) or not np.isfinite(values).all():
            raise ValueError('Each family needs finite, visible training points')
        centers.append(values.mean(0))
        counts.append(len(values))
    return np.asarray(centers), np.asarray(counts)


def hits(peaks, points, half=45, side=224):
    """The scorer's inclusive clipped-square predicate, in recipient coordinates."""
    low = np.clip(peaks[..., None, :] - half, 0, side)
    high = np.clip(peaks[..., None, :] + half, 0, side)
    return np.any(np.all((points >= low) & (points <= high), axis=-1), axis=-1)


def macro(values, attributes, groups=40):
    return float(np.mean([values[attributes == a].mean() for a in range(groups)]))


def bootstrap(values, images, attributes, weights, groups=40):
    """Fixed-attribute macro means; missing groups remain undefined."""
    numerator = np.zeros((weights.shape[1], groups), dtype=float)
    observed = np.zeros_like(numerator)
    np.add.at(numerator, (images, attributes), values)
    np.add.at(observed, (images, attributes), 1)
    denominator = weights @ observed
    valid = (denominator > 0).all(1)
    result = np.full(len(weights), np.nan)
    result[valid] = ((weights @ numerator)[valid] / denominator[valid]).mean(1)
    return result, valid


def summarize(values, data):
    draws, valid = bootstrap(values, data['image'], data['attribute'], data['weights'])
    if not valid.all():
        raise ValueError('Frozen draws must retain all 40 attribute groups')
    return {'mean': macro(values, data['attribute']),
            'ci95': np.quantile(draws, [.025, .975]).tolist()}


def analyze(data):
    train_ids, target_ids = data['training_image_ids'], data['target_image_ids']
    if np.intersect1d(train_ids, target_ids).size:
        raise ValueError('Training calibration and audit recipients overlap')
    prior = data['part_peaks_xy'][data['attribute_part'][data['attribute']]]
    donors = data['donors']
    if not np.all(data['attribute'][donors] == data['attribute'][None]):
        raise ValueError('Donors must preserve concept identity')
    if not np.all(data['image'][donors] != data['image'][None]):
        raise ValueError('Donors cannot be the recipient image')
    if not np.array_equal(prior[donors], np.broadcast_to(prior, prior[donors].shape)):
        raise ValueError('The prior must stay identical across same-concept images')
    s = hits(prior, data['points_xy']).astype(float)
    donor = hits(prior[donors], data['points_xy'][None]).mean(0)
    g = s - donor
    np.testing.assert_array_equal(g, np.zeros_like(g))
    result = {
        'images': int(len(target_ids)), 'queries': int(len(s)), 'attributes': 40,
        'training_images': int(len(train_ids)),
        'calibration_audit_overlap': 0,
        'prior_S': summarize(s, data), 'prior_G': summarize(g, data),
        'exact_uniform_peak_chance': macro(data['chance'], data['attribute']),
        'methods': {},
        'scope': 'Exploratory fixed-prior check on a known cohort; paired intervals condition on training calibration and fixed model/heads/donors.'}
    result['prior_minus_chance'] = summarize(s - data['chance'], data)
    for name in ['native_LocCBM', 'adapted_cam', 'adapted_gradient']:
        model_s = data[name + '__S']
        result['methods'][name] = {
            'model_S': summarize(model_s, data),
            'model_minus_prior': summarize(model_s - s, data)}
    return result
