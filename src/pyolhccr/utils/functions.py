'''Functions for general use.

References: Nazoru input library https://github.com/google/mozc-devices/blob/master/mozc-nazoru/src/nazoru
'''

from typing import List
import numpy as np

def normalize(x: List[np.ndarray], size = (64, 64)):
    """Normalizes position.

    Args:
      x: [[[x, y, ...], ...], ...] List of strokes to normalize the position (x, y).
      size: Size of normalized position.

    Returns:
      x: [[[x', y', ...], ...], ...] List of strokes with the normalized position (x', y').
    """

    max_ = np.max([np.max(stroke[:, :2], axis=0) for stroke in x], axis=0)
    min_ = np.max([np.min(stroke[:, :2], axis=0) for stroke in x], axis=0)
    dist = max_ - min_
    # use fmax to avoid too small distance
    ratio = np.array(size) / np.fmax(dist, np.ones_like(dist) * 0.0001)
    for i in range(len(x)):
        x[i][:, :2] = (x[i][:, :2] - min_) * ratio
    return x