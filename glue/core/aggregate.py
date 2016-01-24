"""
Classes to perform aggregations over cubes
"""

from __future__ import absolute_import, division, print_function

try:
    from itertools import izip
except ImportError:  # python3
    izip = zip

from functools import wraps

import numpy as np

from glue.external.six.moves import range as xrange


def check_empty(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.empty_slice:
            return np.zeros(self.shape) * np.nan
        return func(self, *args, **kwargs)

    return wrapper


class Aggregate(object):

    """
    Collapse >=3D datasets into 2D images, using different
    aggregation methods
    """

    def __init__(self, data, attribute, zax, slc, zlim):
        """
        Parameters
        ----------
        data : :class:`~glue.core.data.Data`
        attribute : :class:`~glue.core.component_id.ComponentID`
        zax : int
            Which axis to collapse over.
        slc : tuple of int
            Describes the
            current 2D slice through the image. Used to
            define the orientation, as well as axis values
            for remaining dimensions of >3D cubes
        zlim : tuple of float
            Float values of [lo, hi), describing the limits
            of the slab to collapse over
        """
        self.data = data
        self.attribute = attribute
        self.zax = zax
        self.slc = slc
        self.zlim = min(zlim), max(zlim)

    @property
    def shape(self):
        """
        The shape of the 2D aggregated array
        """
        s = self.data.shape
        return s[self.slc.index('y')], s[self.slc.index('x')]

    @property
    def empty_slice(self):
        """
        True if the slice is empty
        """
        return self.zlim[0] == self.zlim[1]

    def _subslice(self):
        view = [slice(None, None) for _ in self.data.shape]
        ax_collapse = self.zax
        for i, s in enumerate(self.slc):
            if s not in ['x', 'y'] and i != self.zax:
                view[i] = s
                if i < self.zax:
                    ax_collapse -= 1
        view[self.zax] = slice(*self.zlim)
        return view, ax_collapse

    def _prepare_cube(self, attribute=None):
        view, ax_collapse = self._subslice()
        att = attribute or self.attribute
        cube = self.data[att, view]
        return cube, ax_collapse

    def _iter_slice(self, attribute=None):
        # iterate through the uncollapsed slab one plane at a time
        view, ax_collapse = self._subslice()
        att = attribute or self.attribute

        for z in xrange(*self.zlim):
            view[self.zax] = z
            plane = self.data[att, view]
            yield np.nan_to_num(plane)

    def _iter_slice_index(self):
        """Loop over slices of the target attribute and its world coordinate"""
        att = self.data.get_world_component_id(self.zax)
        loop = izip(self._iter_slice(), self._iter_slice(att))
        return loop

    def _finalize(self, cube):
        if self.slc.index('x') < self.slc.index('y'):
            cube = cube.T
        return cube

    def collapse_using(self, function):
        """
        Produce a collapsed image using a numpy aggregation function
        """
        cube, ax = self._prepare_cube()
        result = function(cube, axis=ax)
        return self._finalize(result)

    def _to_world(self, idx):
        args = [None] * self.data.ndim
        y, x = np.mgrid[:idx.shape[0], :idx.shape[1]]
        for i, s in enumerate(self.slc):
            if s not in ['x', 'y']:
                args[i] = np.ones(idx.size) * s
        args[self.slc.index('y')] = y.ravel()
        args[self.slc.index('x')] = x.ravel()
        args[self.zax] = idx.ravel()
        att = self.data.get_world_component_id(self.zax)
        return self.data[att, args].reshape(idx.shape)

    @staticmethod
    def all_operators():
        return (Aggregate.sum,
                Aggregate.mean,
                Aggregate.max,
                Aggregate.argmax,
                Aggregate.argmin,
                Aggregate.mom1,
                Aggregate.mom2,
                Aggregate.median)

    @staticmethod
    def _mean(cube, axis):
        s = np.nansum(cube, axis)
        ct = np.isfinite(cube).sum(axis)
        return 1. * s / ct

    @check_empty
    def sum(self):
        return self.collapse_using(np.nansum)

    @check_empty
    def mean(self):
        return self.collapse_using(self._mean)

    @check_empty
    def max(self):
        return self.collapse_using(np.nanmax)

    @check_empty
    def median(self):
        # NOTE: nans are treated as infinity in this case
        return self.collapse_using(np.median)

    @check_empty
    def argmax(self):
        """
        Location of peak value, in world coords
        """
        idx = self.collapse_using(np.nanargmax)
        return self._to_world(idx)

    @check_empty
    def argmin(self):
        """
        Location of minimum value, in world coords
        """
        idx = self.collapse_using(np.nanargmin)
        return self._to_world(idx)

    @check_empty
    def mom1(self):
        """
        Intensity-weighted coordinate. Pixel units.
        """
        # build up slice-by-slice, to avoid big temporary cubes
        loop = self._iter_slice_index()
        val, loc = next(loop)

        val = np.maximum(val, 0)
        w, result = val, loc * val
        for val, loc in loop:
            val = np.maximum(val, 0)
            result += val * loc
            w += val
        return self._finalize(result / w)

    @check_empty
    def mom2(self):
        """
        Intensity-weighted coordinate dispersion. Pixel units.
        """
        loop = self._iter_slice_index()
        val, loc = next(loop)

        val = np.maximum(val, 0)
        w, x, x2 = val, val * loc, val * loc * loc
        for val, loc in loop:
            val = np.maximum(val, 0)
            w += val
            x += loc * val
            x2 += loc ** 2 * val

        return self._finalize(np.sqrt(x2 / w - (x / w) ** 2))
