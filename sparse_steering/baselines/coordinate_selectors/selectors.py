"""Select coordinates by magnitude, gradient or indirect effect."""

import math


def vector(values):
    values = [float(v) for v in values]
    if not values or any((not math.isfinite(v) for v in values)):
        raise ValueError("Expected a nonempty finite vector.")
    if math.hypot(*values) == 0:
        raise ValueError("A zero vector cannot define steering.")
    return values


def magnitude_mask(direction, coverage=0.95):
    direction = vector(direction)
    if not math.isfinite(coverage) or not 0 < coverage <= 1:
        raise ValueError("Squared-norm coverage must be in (0, 1].")
    # Scaling first avoids overflow while preserving squared-norm fractions.
    maximum = max(map(abs, direction))
    squared = [(v / maximum) ** 2 for v in direction]
    total = math.fsum(squared)
    order = sorted(range(len(direction)), key=lambda i: (-squared[i], i))
    mask = [False] * len(direction)
    kept = []
    for i in order:
        if direction[i] == 0:
            break
        kept.append(squared[i])
        mask[i] = True
        if math.fsum(kept) >= coverage * total:
            break
    if math.fsum(kept) < coverage * total:
        raise ValueError("Coverage is not numerically attainable.")
    return mask


def cheng_masks(direction, indirect_effect, threshold=1.0):
    direction, effect = (vector(direction), vector(indirect_effect))
    if len(direction) != len(effect):
        raise ValueError("Direction and IE widths differ.")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Threshold must be finite and nonnegative.")
    s_norm, ie_norm = (math.hypot(*direction), math.hypot(*effect))
    if any((s == 0 and e != 0 for s, e in zip(direction, effect))):
        raise ValueError("IE must vanish wherever the steering direction is zero.")
    # Normalize the two vectors separately before comparing absolute effects.
    gradient = [
        s != 0 and abs(e) / ie_norm >= threshold * abs(s) / s_norm
        for s, e in zip(direction, effect)
    ]
    # The IE comparison uses the same cardinality as the gradient selector.
    count = sum(gradient)
    if count == 0:
        raise ValueError("Threshold removes all coordinates; do not silently change it.")
    eligible = [i for i, s in enumerate(direction) if s != 0]
    order = sorted(eligible, key=lambda i: (-abs(effect[i]), i))
    ie = [False] * len(direction)
    for i in order[:count]:
        ie[i] = True
    return gradient, ie


def masked_direction(direction, mask):
    direction = vector(direction)
    if len(direction) != len(mask) or any((type(v) is not bool for v in mask)):
        raise ValueError("Expected a boolean mask with the direction width.")
    return vector([v if keep else 0.0 for v, keep in zip(direction, mask)])


def fits(direction, indirect_effect, coverage=0.95, threshold=1.0):
    direction = vector(direction)
    magnitude = magnitude_mask(direction, coverage)
    gradient, ie = cheng_masks(direction, indirect_effect, threshold)
    norm = math.hypot(*direction)
    result = {}
    for name, mask in [("magnitude", magnitude), ("gradient", gradient), ("ie", ie)]:
        masked = masked_direction(direction, mask)
        result[name] = dict(
            direction=masked,
            mask=mask,
            support_count=sum(mask),
            width=len(mask),
            retained_fraction=sum(mask) / len(mask),
            squared_norm_coverage=(math.hypot(*masked) / norm) ** 2,
            vector_renormalized=False,
            activation_rescaling=False,
        )
    return result
