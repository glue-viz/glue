import logging

import numpy as np
from ..core.decorators import memoize


def get_extent(view, transpose=False):
    sy, sx = [s for s in view if isinstance(s, slice)]
    if transpose:
        return (sy.start, sy.stop, sx.start, sx.stop)
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


@memoize
def fast_limits(data, component, plo, phi):
    """Quickly estimate percentiles in a data object,
    using a downsampled version

    :param data: data object to inspect
    :param component: Component to inspect
    :param plo: Lo percentile
    :param phi: High percentile

    :rtype: Tuple of floats. Approximate values of each percentile in
            data[component]
    """
    try:
        from scipy import stats
    except ImportError:
        raise ImportError("Scale clipping requires SciPy")

    shp = data.shape
    view = tuple([slice(None, None, max(s / 50, 1)) for s in shp])
    values = data[component, view]
    limits = (-np.inf, np.inf)
    lo = stats.scoreatpercentile(values.flat, plo, limit=limits)
    hi = stats.scoreatpercentile(values.flat, phi, limit=limits)
    return lo, hi


def visible_limits(artists, axis):
    """Determines the data limits for the data in a set of artists

    Ignores non-visible artists

    Assumes each artist as a get_data method wich returns a tuple of x,y

    :param artists: An iterable collection of artists
    :param axis: Which axis to compute. 0=xaxis, 1=yaxis

    :rtype: A tuple of min, max for the requested axis, or None if
            no data present
    """
    data = []
    for art in artists:
        if not art.visible:
            continue
        xy = art.get_data()
        assert isinstance(xy, tuple)
        val = xy[axis]
        if val.size > 0:
            data.append(xy[axis])

    if len(data) == 0:
        return
    data = np.hstack(data)
    if data.size == 0:
        return

    data = data[np.isfinite(data)]
    if data.size == 0:
        return

    lo, hi = np.nanmin(data), np.nanmax(data)
    if not np.isfinite(lo):
        return

    return lo, hi
