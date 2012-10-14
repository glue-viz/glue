import logging


def get_extent(view):
    sy, sx = [s for s in view if isinstance(s, slice)]
    return (sx.start, sx.stop, sy.start, sy.stop)


def view_cascade(data, view):
    """ Return a set of views progressively zoomed out of input at roughly
    constant pixel count

    :param data: Data object to view
    :param view: Original view into data

    :rtype: tuple of views
    """
    shp = data.shape
    v2 = list(view)
    logging.debug("image shape: %s, view: %s", shp, view)

    #choose stride length that roughly samples entire image
    #at roughly the same pixel count
    step = max(shp[i - 1] * v.step / max(v.stop - v.start, 1)
               for i, v in enumerate(view) if isinstance(v, slice))
    step = max(step, 1)

    for i, v in enumerate(v2):
        if not(isinstance(v, slice)):
            continue
        v2[i] = slice(0, shp[i - 1], step)

    return tuple(v2), view
