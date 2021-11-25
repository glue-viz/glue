"""
Matplotlib artist for fixed resolution buffers.
"""

from mpl_scatter_density.base_image_artist import BaseImageArtist

import numpy as np

from glue.utils import color2rgb


class FRBArtist(BaseImageArtist):

    def __init__(self, ax, **kwargs):
        self._array_maker = None
        super(FRBArtist, self).__init__(ax, update_while_panning=False,
                                        array_func=self.array_func_wrapper,
                                        **kwargs)

    def array_func_wrapper(self, bins=None, range=None):
        if self._array_maker is None:
            return np.array([[np.nan]])
        else:
            ny, nx = bins
            (ymin, ymax), (xmin, xmax) = range
            bounds = [(ymin, ymax, ny), (xmin, xmax, nx)]
            array = self._array_maker(bounds)
            if array is None:
                return np.array([[np.nan]])
            else:
                return array

    def set_array_maker(self, array_maker):
        self._array_maker = array_maker

    def invalidate_cache(self):
        self.stale = True
        self.set_visible(True)


def imshow(axes, array_maker, aspect=None, vmin=None, vmax=None, color=None, **kwargs):
    """
    Similar to matplotlib's imshow command, but produces a FRBArtist
    """

    axes.set_aspect(aspect)

    im = FRBArtist(axes, **kwargs)

    if color:

        def wrapper(bounds=None):

            # Get original array
            mask = array_maker(bounds=bounds)

            # Convert to RGBA array"
            r, g, b = color2rgb(color)
            mask = np.dstack((r * mask, g * mask, b * mask, mask * .5))
            mask = (255 * mask).astype(np.uint8)

            return mask

        im.set_array_maker(wrapper)

    else:

        im.set_array_maker(array_maker)

    axes._set_artist_props(im)

    if im.get_clip_path() is None:
        # image does not already have clipping set, clip to axes patch
        im.set_clip_path(axes.patch)

    if vmin is not None or vmax is not None:
        im.set_clim(vmin, vmax)

    axes.add_image(im)

    return im
