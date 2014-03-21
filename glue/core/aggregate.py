"""
Classes to perform aggregations over cubes
"""
try:
    from itertools import izip
except ImportError:  # python3
    izip = zip

import numpy as np


class Aggregate(object):

    def __init__(self, data, attribute, zax, slc, zlim):
        self.data = data
        self.attribute = attribute
        self.zax = zax
        self.slc = slc
        self.zlim = zlim

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

    def sum(self):
        return self.collapse_using(np.nansum)

    def mean(self):
        return self.collapse_using(self._mean)

    def max(self):
        return self.collapse_using(np.nanmax)

    def median(self):
        # NOTE: nans are treated as infinity in this case
        return self.collapse_using(np.median)

    def argmax(self):
        idx = self.collapse_using(np.nanargmax)
        return self._to_world(idx)

    def argmin(self):
        idx = self.collapse_using(np.nanargmin)
        return self._to_world(idx)

    def mom1(self):
        # build up slice-by-slice, to avoid big temporary cubes
        loop = self._iter_slice_index()
        val, loc = next(loop)
        w, result = val, loc * val
        for val, loc in loop:
            result += val * loc
            w += val
        return self._finalize(result / w)

    def mom2(self):
        loop = self._iter_slice_index()
        val, loc = next(loop)
        w, x, x2 = val, val * loc, val * loc * loc
        for val, loc in loop:
            w += val
            x += loc * val
            x2 += loc ** 2 * val

        return self._finalize(np.sqrt(x2 / w - (x / w) ** 2))
