"""
This module implements polygon-square intersection using matplotlib. It is
twice as fast as Shapely for this specific case and avoids requiring another
dependency.
"""

import numpy as np

from matplotlib.path import Path
from matplotlib.transforms import Bbox


def square_polygon_intersection(xmin, xmax, ymin, ymax, x, y):
    poly = Path(list(zip(x, y)))
    box = Bbox([[xmin, ymin], [xmax, ymax]])
    try:
        clipped_poly = poly.clip_to_bbox(box)
    except ValueError:
        return [], []
    else:
        return clipped_poly.vertices[:, 0], clipped_poly.vertices[:, 1]


def polygon_area(x, y):
    x1 = x
    x2 = np.roll(x, -1)
    y1 = y
    y2 = np.roll(y, -1)
    return abs(0.5 * np.sum(x1 * y2 - x2 * y1))


def square_polygon_overlap_area(xmin, xmax, ymin, ymax, x, y):
    x, y = square_polygon_intersection(xmin, xmax, ymin, ymax, x, y)
    if len(x) == 0:
        return 0.
    else:
        return polygon_area(x, y)
