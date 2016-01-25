from __future__ import absolute_import, division, print_function

import numpy as np

__all__ = ['polygon_line_intersections']


def polygon_line_intersections(px, py, xval=None, yval=None):
    """
    Find all the segments of intersection between a polygon and an infinite
    horizontal/vertical line.

    The polygon is assumed to be closed

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

    # TODO: vectorize this later, but need to get the logic for corner cases
    # right first.

    vertex = px == xval

    points = []

    for i in range(len(px) - 1):

        if vertex[i] and vertex[i+1]:

            # Special case where both vertices are on the line

            points.append(py[i])
            points.append(py[i+1])

        elif vertex[i]:

            # First vertex is on the line

            points.append(py[i])

        elif (px[i] < xval and px[i+1] > xval) or (px[i+1] < xval and px[i] > xval):

            y = py[i] + (py[i+1] - py[i]) * (xval - px[i]) / (px[i+1] - px[i])

            points.append(y)

    # Make unique and sort
    points = sorted(set(points))

    # Make into a list of tuples
    return list(zip(points[::2], points[1::2]))
