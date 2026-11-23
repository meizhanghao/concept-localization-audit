"""Peak/window scoring primitives; geometry stays with the recipient image."""
import numpy as np


def keypoint_hit(peaks, points, half_width=45, side=224):
    """CUB's inclusive, clipped peak-centered square/keypoint predicate.

    Coordinates are (x, y). A hit occurs when any valid target keypoint falls
    within the square centered on the supplied peak. Cached invalid points are
    (-1, -1), so they cannot satisfy the clipped window predicate.
    """
    low = np.maximum(peaks[..., None, :] - half_width, 0)
    high = np.minimum(peaks[..., None, :] + half_width, side)
    return ((points >= low) & (points <= high)).all(-1).any(-1)
