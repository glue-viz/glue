from scipy import ndimage
from matplotlib import _cntr
import numpy as np

def relim(lo, hi, log=False):
    x, y = lo, hi
    if log:
        if lo < 0:
            x = 1e-5
        if hi < 0:
            y = 1e5
    return (x, y)

def file_format(filename):
    if filename.find('.') == -1:
        return ''
    if filename.lower().endswith('.gz'):
        result = filename.lower().rsplit('.', 2)[1]
    else:
        result = filename.lower().rsplit('.', 1)[1]
    return result

def point_contour(x, y, data):
    """Calculate the contour that passes through (x,y) in data

    :param x: x location
    :param y: y location
    :param data: 2D image
    :type data: ndarray

    Returns:

       * A (nrow, 2column) numpy array. The two columns give the x and
         y locations of the contour vertices
    """
    inten = data[y, x]
    labeled, nr_objects = ndimage.label(data >= inten)
    z = data * (labeled == labeled[y, x])
    y, x = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    cnt = _cntr.Cntr(x, y, z)
    xy = cnt.trace(inten)
    if not xy:
        return None
    xy = xy[0]
    return xy