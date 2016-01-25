def polygon_line_intersections(px, py, xval=None, yval=None):
    """
    Find all the segments of intersection between a polygon and an infinite
    horizontal/vertical line.

    The polygon is assumed to be closed.

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

    # Find sections that intersect with the infinite vertical line
    keep = (px[1:] > xval) == (px[:-1] < xval)

    # Extract intersecting segments
    x1 = px[:-1][keep]
    x2 = px[1:][keep]
    y1 = py[:-1][keep]
    y2 = py[1:][keep]

    # Determine intersection points
    y = y1 + (y2 - y1) * (xval - x1) / (x2 - x1)

    # Sort intersection points and group into pairs
    y.sort()
    pairs = y.reshape((-1, 2)).tolist()

    return pairs
