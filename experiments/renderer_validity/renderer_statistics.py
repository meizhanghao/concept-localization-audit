"""NumPy-only statistics for the explicitly declared renderer intervention.

The bootstrap repeats images. Weighted midranks reproduce the ranks obtained
by physically repeating each image, including ties; heads and donors stay fixed.
No outcome here is a claim about a natural-world semantic intervention.
"""
import numpy as np

PARTS = ['beak', 'eye', 'foot', 'tail', 'wing']
MODELS = ['resnet18_seed0', 'resnet18_seed1', 'resnet50_seed2']
METHODS = ['cam', 'gradient']
OFFSETS = np.array([0, 5, 9, 14, 24, 31])


def interval(draws):
    draws = np.asarray(draws)
    valid = np.isfinite(draws)
    return np.quantile(draws[valid], [.025, .975]).tolist() if valid.any() else None


def estimate(point, draws):
    fraction = float(np.isfinite(draws).mean())
    return {'estimate': float(point) if np.isfinite(point) else None,
            'ci95': interval(draws),
            'valid_bootstrap_fraction': fraction,
            'interval_scope': ('Image-cluster percentile interval, fixed heads/donors.'
                               if fraction == 1 else
                               'Conditional on defined bootstrap statistics; NOT an unconditional 95% interval.')}


def weighted_midranks(values, weights):
    """Midranks of a multiset, up to an irrelevant common additive constant."""
    order = np.argsort(values, kind='stable')
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1])+1]
    counts = np.add.reduceat(weights[:, order], starts, axis=1)
    group = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(values)]))
    ranks = (counts.cumsum(1)-counts/2)[:, group]
    return ranks[:, np.argsort(order)]


def weighted_spearman(x, y, weights):
    rx = weighted_midranks(x, weights)
    ry = weighted_midranks(y, weights)
    total = weights.sum(1, keepdims=True)
    mx = (weights*rx).sum(1, keepdims=True)/np.maximum(total, 1)
    my = (weights*ry).sum(1, keepdims=True)/np.maximum(total, 1)
    rx -= mx
    ry -= my
    denominator = np.sqrt((weights*rx**2).sum(1)*(weights*ry**2).sum(1))
    return np.divide((weights*rx*ry).sum(1), denominator,
                     out=np.full(len(weights), np.nan), where=denominator > 0)


def macro_correlation(x, y, eligible, weights):
    all_weights = np.vstack([np.ones(len(x)), weights])
    correlations = np.stack([
        weighted_spearman(x[:, part], y[:, part],
                          all_weights*eligible[:, part][None])
        for part in range(5)], axis=1)
    # Do not hide an undefined part by silently averaging the other parts.
    macro = correlations.mean(1)
    record = estimate(macro[0], macro[1:])
    record['per_part'] = [float(v) if np.isfinite(v) else None
                          for v in correlations[0]]
    return record, macro[1:], correlations


def probability(logits):
    result = np.empty_like(logits, dtype=float)
    for a, b in zip(OFFSETS[:-1], OFFSETS[1:]):
        value = logits[..., a:b].astype(float)
        exponential = np.exp(value-value.max(-1, keepdims=True))
        result[..., a:b] = exponential/exponential.sum(-1, keepdims=True)
    return result


def scores_from_peaks(data, rows, donors):
    side = int(data['side'])
    windows = np.unpackbits(data['windows_packed'][rows], axis=-1,
                            count=side*side).astype(bool)
    peaks = data['peaks'][:, :, :, rows]
    position = peaks[..., 0]*side+peaks[..., 1]
    own = windows[np.arange(len(rows))[None, None, None, :, None],
                  np.arange(5)[None, None, None, None, :], position].astype(float)
    transferred = position[:, :, :, donors, np.arange(5)]
    donor_hits = windows[np.arange(len(rows))[None, None, None, None, :, None],
                        np.arange(5)[None, None, None, None, None, :],
                        transferred].mean(3)
    gain = own-donor_hits
    return np.stack([own[:, :, 0], gain[:, :, 0],
                     gain[:, :, 0]-gain[:, :, 1:].mean(2)], axis=2)


def selected_response(scores, response, eligible):
    """One part per image, uniform over exact score ties; no fitted selector."""
    counts = eligible.sum(1)
    assert (counts > 0).all()
    valid_score = np.where(eligible, scores, -np.inf)
    ties = eligible & np.isclose(valid_score, valid_score.max(1, keepdims=True),
                                 rtol=0, atol=1e-12)
    chosen = (response*ties).sum(1)/ties.sum(1)
    return chosen, ties, counts


def selection_utility(scores, response, eligible, weights):
    chosen, ties, counts = selected_response(scores, response, eligible)
    uniform = (response*eligible).sum(1)/counts
    oracle = np.where(eligible, response, -np.inf).max(1)
    gain = chosen-uniform
    regret = oracle-chosen
    return {'selected_minus_uniform': estimate(gain.mean(), weights@gain/len(gain)),
            'regret': estimate(regret.mean(), weights@regret/len(regret)),
            'uniform_regret': float((oracle-uniform).mean()),
            'mean_tie_count': float(ties.sum(1).mean()),
            'mean_available_parts': float(counts.mean())}


def analyze(data):
    query = data['queries']
    alternative = data['alternatives']
    logits = data['logits'].astype(float)
    removed = data['deleted_logits'].astype(float)
    before = np.take_along_axis(logits, query[None], -1)-np.take_along_axis(
        logits, alternative[None], -1)
    after = np.take_along_axis(removed, query[None, :, None, :], -1)-np.take_along_axis(
        removed, alternative[None, :, None, :], -1)
    full_response = before[:, :, None, :]-after
    concept_response = np.diagonal(full_response, axis1=2, axis2=3).copy()
    specific_response = concept_response-(full_response.sum(2)-concept_response)/4
    p, pd = probability(logits[0]), probability(removed[0])
    readout_weight, readout_bias = data['readout_weight'], data['readout_bias']
    class_logits = p@readout_weight.T+readout_bias
    class_logits[np.arange(len(query)), data['classes']] = -np.inf
    rival = class_logits.argmax(1)
    contrast_weight = readout_weight[data['classes']]-readout_weight[rival]
    class_response = np.einsum('nc,nkc->nk', contrast_weight, p[:, None]-pd)
    output = {'scope': 'Exploratory rendered-deletion response association; fixed models, three Gaussian heads and donor schedules; not natural-world ACE.',
              'models': MODELS, 'parts': PARTS, 'cohorts': {}}
    for cohort in ['original200', 'complement300']:
        rows = data[cohort+'_rows']
        weights = data[cohort+'_weights'].astype(float)
        scores = scores_from_peaks(data, rows, data[cohort+'_donors'])
        entry = {'images': len(rows), 'queries': len(rows)*5,
                 'unmasked_queries_by_part': np.full(5, len(rows)).tolist(),
                 'visible_queries_by_part': data['visible'][rows].sum(0).tolist(),
                 'exactly_unchanged_deletions_by_part': data['unchanged'][rows].sum(0).tolist(),
                 'mean_pixel_change': data['pixel_mean_l1'][rows].mean(0).tolist(),
                 'analyses': {}}
        for support in ['all', 'visible']:
            eligible = np.ones((len(rows), 5), bool) if support == 'all' else data['visible'][rows]
            records = {}
            differences, difference_draws = [], []
            for model_index, model in enumerate(MODELS):
                model_record = {}
                for method_index, method in enumerate(METHODS):
                    s = scores[model_index, method_index]
                    record, score_draws = {}, {}
                    for target, response in [('concept_margin', concept_response),
                                              ('specific_margin', specific_response)]:
                        record[target] = {}
                        for i, score_name in enumerate(['S', 'G', 'deltaG']):
                            summary, draws, per_part = macro_correlation(
                                s[i], response[model_index, rows], eligible, weights)
                            record[target][score_name] = summary
                            score_draws[target+'_'+score_name] = draws
                            score_draws[target+'_'+score_name+'_per_part'] = per_part
                        a, b = (record[target][name]['estimate'] for name in ['deltaG', 'S'])
                        delta = a-b if a is not None and b is not None else np.nan
                        paired = score_draws[target+'_deltaG']-score_draws[target+'_S']
                        record[target]['deltaG_minus_S'] = estimate(delta, paired)
                        per_part_difference = (score_draws[target+'_deltaG_per_part']-
                                               score_draws[target+'_S_per_part'])
                        record[target]['deltaG_minus_S']['per_part'] = {
                            part: estimate(per_part_difference[0, p], per_part_difference[1:, p])
                            for p, part in enumerate(PARTS)}
                        if target == 'concept_margin':
                            differences.append(delta)
                            difference_draws.append(paired)
                    if model_index == 0:
                        record['class_margin_selection'] = {
                            name: selection_utility(s[i], class_response[rows], eligible, weights)
                            for i, name in enumerate(['S', 'G', 'deltaG'])}
                        selected_s = selected_response(s[0], class_response[rows], eligible)[0]
                        selected_delta = selected_response(s[2], class_response[rows], eligible)[0]
                        change = selected_delta-selected_s
                        record['selection_deltaG_minus_S'] = estimate(
                            change.mean(), weights@change/len(change))
                    model_record[method] = record
                records[model] = model_record
            draws = np.stack(difference_draws, axis=1)
            point = np.asarray(differences)
            estimable = np.isfinite(point)
            valid = np.isfinite(draws[:, estimable]).all(1)
            simultaneous = [None]*len(point)
            conditional_bands = [None]*len(point)
            if estimable.any() and valid.any():
                selected = draws[valid][:, estimable]
                se = selected.std(0, ddof=1)
                assert (se > 0).all()
                quantile = np.quantile(np.max(abs((selected-point[estimable])/se), axis=1), .95)
                bounds = np.stack([point[estimable]-quantile*se, point[estimable]+quantile*se], 1)
                for i, bound in zip(np.flatnonzero(estimable), bounds):
                    conditional_bands[i] = bound.tolist()
                    if valid.all() and estimable.all():
                        simultaneous[i] = bound.tolist()
            entry['analyses'][support] = {
                'models': records,
                'six_comparison_order': [m+'__'+e for m in MODELS for e in METHODS],
                'simultaneous_deltaG_minus_S_ci95': simultaneous,
                'diagnostic_defined_draw_bands': conditional_bands,
                'simultaneous_status': ('available' if valid.all() and estimable.all()
                                        else 'withheld: degenerate statistic in one or more comparisons/draws'),
                'estimable_comparisons': estimable.tolist(),
                'all_six_comparisons_estimable': bool(estimable.all()),
                'simultaneous_valid_fraction': float(valid.mean()),
                'simultaneous_method': 'Centered maximum-standardized image bootstrap over six declared comparisons. Simultaneous inferential claim withheld if any comparison/draw is undefined. Defined-draw bands are diagnostics only; no part is silently omitted.'}
        output['cohorts'][cohort] = entry
    return output
