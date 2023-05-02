import numpy as np

from glue.utils import unbroadcast

__all__ = ['points_inside_poly', 'polygon_line_intersections', 'floodfill', 'rotation_matrix_2d']


def rotation_matrix_2d(alpha):
    """
    Return rotation matrix for angle alpha around origin.

    Parameters
    ----------
    alpha : float
        Rotation angle in radian, increasing for anticlockwise rotation.
    """
    if np.asarray(alpha).ndim > 0:
        # In principle this works on an array as well; would have to return matrix.T then
        raise ValueError("Only scalar input for angle accepted")

    return np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])


def points_inside_poly(x, y, vx, vy):
    """
    Test if coordinates ``x``, ``y`` fall inside polygon of vertices ``vx``, ``vy``.

    Parameters
    ----------
    x, y : `~numpy.ndarray`
        Coordinates of the points to test
    vx, vy : `~numpy.ndarray`
        The vertices of the polygon

    Returns
    -------
    contains : `~numpy.ndarray` of bool
        Array indicating whether each coordinate pair is inside the polygon.
    """

    if x.dtype.kind == 'M' and vx.dtype.kind == 'M':
        vx = vx.astype(x.dtype).astype(float)
        x = x.astype(float)

    if y.dtype.kind == 'M' and vy.dtype.kind == 'M':
        vy = vy.astype(y.dtype).astype(float)
        y = y.astype(float)

    original_shape = x.shape

    x = unbroadcast(x)
    y = unbroadcast(y)

    x = x.astype(float)
    y = y.astype(float)

    x, y = np.broadcast_arrays(x, y)

    reduced_shape = x.shape

    x = x.flat
    y = y.flat

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

    good = np.isfinite(x) & np.isfinite(y)
    inside[keep][~good] = False

    inside = inside.reshape(reduced_shape)
    inside = np.broadcast_to(inside, original_shape)

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


def floodfill(data, start_coords, threshold):

    from scipy.ndimage import label

    # Determine value at the starting coordinates
    value = data[start_coords]

    # Determine all pixels that match
    mask = (data > value * (2 - threshold)) & (data < value * threshold)

    # Determine all individual chunks
    labels, num_features = label(mask)

    mask = labels == labels[start_coords]

    return mask
