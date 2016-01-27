"""
This file implements a matplotlib Normalize object
which mimics the functionality of scaling functions in ds9

The transformation from data values to normalized (0-1) display
intensities are as follows:

- Data to normal:
   y = clip( (x - vmin) / (vmax - vmin), 0, 1)
- normal to warped: Apply a monotonic, non-linear scaling, that preserves
  the endpoints
- warped to greyscale:
  y = clip((x - bias) * contrast + 0.5, 0, 1)
"""

# implementation details
# The relevant ds9 code is located at saotk/frame/colorscale.C and
# saotk/colorbar/colorbar.C
#
# As much as possible, we use verbose but inplace ufuncs to minimize
# temporary arrays

from __future__ import absolute_import, division, print_function

import numpy as np
from matplotlib.colors import Normalize

from glue.utils import fast_limits


def norm(x, vmin, vmax):
    """
    Linearly scale data between [vmin, vmax] to [0, 1]. Clip outliers
    """
    result = (x - 1.0 * vmin)
    result = np.divide(result, vmax - vmin, out=result)
    result = np.clip(result, 0, 1, out=result)
    return result


def cscale(x, bias, contrast):
    """
    Apply bias and contrast scaling in-place.

    Parameters
    ----------
    x : array
        The input values, scaled to the [0:1] range.
    bias : float
        The bias to apply to the data
    contrast : float
        The contrast to apply to the data
    """
    x = np.subtract(x, bias, out=x)
    x = np.multiply(x, contrast, out=x)
    x = np.add(x, 0.5, out=x)
    x = np.clip(x, 0, 1, out=x)
    return x


def linear_warp(x, vmin, vmax, bias, contrast):
    return cscale(norm(x, vmin, vmax), bias, contrast)


def log_warp(x, vmin, vmax, bias, contrast, exp=1000.0):
    black = x < vmin
    x = norm(x, vmin, vmax)
    x = np.multiply(exp, x, out=x)
    # sidestep numpy bug that masks log(1)
    # when out is provided
    x = np.add(x, 1.001, out=x)
    x = np.log(x, out=x)
    x = np.divide(x, np.log(exp + 1.0), out=x)
    x = cscale(x, bias, contrast)
    return x


def pow_warp(x, vmin, vmax, bias, contrast, exp=1000.0):
    x = norm(x, vmin, vmax)
    x = np.power(exp, x, out=x)
    x = np.subtract(x, 1, out=x)
    x = np.divide(x, exp - 1)
    x = cscale(x, bias, contrast)
    return x


def sqrt_warp(x, vmin, vmax, bias, contrast):
    x = norm(x, vmin, vmax)
    x = np.sqrt(x, out=x)
    x = cscale(x, bias, contrast)
    return x


def squared_warp(x, vmin, vmax, bias, contrast):
    x = norm(x, vmin, vmax)
    x = np.power(x, 2, out=x)
    x = cscale(x, bias, contrast)
    return x


def asinh_warp(x, vmin, vmax, bias, contrast):
    x = norm(x, vmin, vmax)
    x = np.divide(np.arcsinh(np.multiply(x, 10, out=x), out=x), 3, out=x)
    x = cscale(x, bias, contrast)
    return x

warpers = dict(linear=linear_warp,
               log=log_warp,
               sqrt=sqrt_warp,
               power=pow_warp,
               squared=squared_warp,
               arcsinh=asinh_warp)


# for mpl <= 1.1, Normalize is an old-style class
# explicitly inheriting from object allows property to work
class DS9Normalize(Normalize, object):

    def __init__(self):
        super(DS9Normalize, self).__init__()
        self.stretch = 'linear'
        self.bias = 0.5
        self.contrast = 1.0
        self.clip_lo = 5.
        self.clip_hi = 95.

    @property
    def stretch(self):
        return self._stretch

    @stretch.setter
    def stretch(self, value):
        if value not in warpers:
            raise ValueError("Invalid stretch: %s\n Valid options are: %s" %
                             (value, warpers.keys()))
        self._stretch = value

    def update_clip(self, image):
        vmin, vmax = fast_limits(image, self.clip_lo, self.clip_hi)
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, value, clip=False):
        # XXX ignore clip

        self.autoscale_None(value)  # set vmin, vmax if unset
        inverted = self.vmax <= self.vmin

        hi, lo = max(self.vmin, self.vmax), min(self.vmin, self.vmax)

        warp = warpers[self.stretch]
        result = warp(value, lo, hi, self.bias, self.contrast)

        if inverted:
            result = np.subtract(1, result, out=result)

        result = np.ma.MaskedArray(result, copy=False)

        return result

    def __gluestate__(self, context):
        return dict(vmin=self.vmin, vmax=self.vmax, clip_lo=self.clip_lo,
                    clip_hi=self.clip_hi, stretch=self.stretch, bias=self.bias,
                    contrast=self.contrast)

    @classmethod
    def __setgluestate__(cls, rec, context):
        result = cls()
        for k, v in rec.items():
            setattr(result, k, v)
        return result
