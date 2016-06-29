from __future__ import absolute_import, division, print_function

import numpy as np

__all__ = ['points_inside_poly', 'polygon_line_intersections']


def points_inside_poly(x, y, vx, vy):

    from matplotlib.path import Path
    p = Path(np.column_stack((vx, vy)))

    keep = ((x >= np.min(vx)) &
            (x <= np.max(vx)) &
            (y >= np.min(vy)) &
            (y <= np.max(vy)))

    inside = np.zeros(len(x), bool)

    x = x[keep]
    y = y[keep]

    coords = np.column_stack((x, y))

    inside[keep] = p.contains_points(coords).astype(bool)

    return inside


def polygon_line_intersections(px, py, xval=None, yval=None):
    """
    Find all the segments of intersection between a polygon and an infinite
    horizontal/vertical line.

    The polygon is assumed to be closed. Due to numerical precision, the
    behavior at the edges of polygons is not always predictable, i.e. a point
    on the edge of a polygon may be considered inside or outside the polygon.

    Parameters
    ----------
    px, py : `~numpy.ndarray`
        The vertices of the polygon
    xval : float, optional
        The x coordinate of the line (for vertical lines). This should only be
        specified if yval is not specified.
    yval : float, optional
        The y coordinate of the line (for horizontal lines). This should only be
        specified if xval is not specified.

    Returns
    -------
    segments : list
        A list of segments given as tuples of coordinates along the line.
    """

    if xval is not None and yval is not None:
        raise ValueError("Only one of xval or yval should be specified")
    elif xval is None and yval is None:
        raise ValueError("xval or yval should be specified")
    if yval is not None:
        return polygon_line_intersections(py, px, xval=yval)

    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)

    # Make sure that the polygon is closed
    if px[0] != px[-1] or py[0] != py[-1]:
        px = np.hstack([px, px[0]])
        py = np.hstack([py, py[0]])

    # For convenience
    x1, x2 = px[:-1], px[1:]
    y1, y2 = py[:-1], py[1:]

    # Vertices that intersect
    keep1 = (px == xval)
    points1 = py[keep1]

    # Segments (excluding vertices) that intersect
    keep2 = ((x1 < xval) & (x2 > xval)) | ((x2 < xval) & (x1 > xval))
    points2 = (y1 + (y2 - y1) * (xval - x1) / (x2 - x1))[keep2]

    # Make unique and sort
    points = np.array(np.sort(np.unique(np.hstack([points1, points2]))))

    # Because of various corner cases, we don't actually know which pairs of
    # points are inside the polygon, so we check this using the mid-points
    ymid = 0.5 * (points[:-1] + points[1:])
    xmid = np.repeat(xval, len(ymid))
    keep = points_inside_poly(xmid, ymid, px, py)

    segments = list(zip(points[:-1][keep], points[1:][keep]))

    return segments
