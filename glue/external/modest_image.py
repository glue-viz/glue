"""
Modification of Chris Beaumont's mpl-modest-image package to allow the use of
set_extent.
"""
from __future__ import print_function, division

import matplotlib
rcParams = matplotlib.rcParams

import matplotlib.colors as mcolors
from mpl_scatter_density.base_image_artist import BaseImageArtist

import numpy as np


class ModestImage(BaseImageArtist):

    def __init__(self, ax, **kwargs):
        self._array_maker = None
        super(ModestImage, self).__init__(ax, update_while_panning=False,
                                          array_func=self.array_func_wrapper,
                                          **kwargs)

    def array_func_wrapper(self, bins=None, range=None):
        if self._array_maker is None:
            return np.array([[np.nan]])
        else:
            ny, nx = bins
            (ymin, ymax), (xmin, xmax) = range
            bounds = [(ymin, ymax, ny), (xmin, xmax, nx)]
            return self._array_maker.get_array(bounds)

    def set_array_maker(self, array_maker):
        self._array_maker = array_maker

    def invalidate_cache(self):
        self.stale = True
        self.set_visible(True)


def imshow(axes, X, cmap=None, norm=None, aspect=None,
           interpolation=None, alpha=None, vmin=None, vmax=None,
           origin=None, extent=None, shape=None, filternorm=1,
           filterrad=4.0, imlim=None, resample=None, url=None, **kwargs):
    """Similar to matplotlib's imshow command, but produces a ModestImage

    Unlike matplotlib version, must explicitly specify axes
    """
    if norm is not None:
        assert(isinstance(norm, mcolors.Normalize))
    if aspect is None:
        aspect = rcParams['image.aspect']
    axes.set_aspect(aspect)
    im = ModestImage(axes, cmap=cmap, norm=norm, interpolation=interpolation,
                     origin=origin, extent=extent, filternorm=filternorm,
                     filterrad=filterrad, resample=resample, **kwargs)

    im.set_array_maker(X)
    im.set_alpha(alpha)
    axes._set_artist_props(im)

    if im.get_clip_path() is None:
        # image does not already have clipping set, clip to axes patch
        im.set_clip_path(axes.patch)

    # if norm is None and shape is None:
    #    im.set_clim(vmin, vmax)
    if vmin is not None or vmax is not None:
        im.set_clim(vmin, vmax)
    # elif norm is None:
    #     im.autoscale_None()

    im.set_url(url)

    # update ax.dataLim, and, if autoscaling, set viewLim
    # to tightly fit the image, regardless of dataLim.
    if extent is not None:
        im.set_extent(extent)

    axes.images.append(im)

    def remove(h):
        axes.images.remove(h)

    im._remove_method = remove

    return im
