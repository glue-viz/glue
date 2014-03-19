"""
Classes to perform aggregations over cubes
"""
from itertools import izip

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
            yield plane

    def _finalize(self, cube):
        if self.slc.index('x') < self.slc.index('y'):
            cube = cube.T
        return cube

    def collapse_using(self, function):
        cube, ax = self._prepare_cube()
        result = function(cube, axis=ax)
        return self._finalize(result)

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

    def mom1(self):
        # build up slice-by-slice, to avoid big temporary cubes
        w, result = None, None
        att = self.data.get_world_component_id(self.zax)

        for val, loc in izip(self._iter_slice(),
                             self._iter_slice(att)):

            val = np.nan_to_num(val)
            if w is None:
                w, result = val, loc * val
            else:
                result += val * loc
                w += val
        return result / w

        return self._finalize(result)
