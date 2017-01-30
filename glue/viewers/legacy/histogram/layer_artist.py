from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractproperty, abstractmethod

import numpy as np

from glue.external import six
from glue.core.exceptions import IncompatibleAttribute
from glue.core.layer_artist import MatplotlibLayerArtist, ChangedTrigger

__all__ = ['HistogramLayerArtist']


@six.add_metaclass(ABCMeta)
class HistogramLayerBase(object):
    lo = abstractproperty()     # lo-cutoff for bin counting
    hi = abstractproperty()     # hi-cutoff for bin counting
    nbins = abstractproperty()  # number of bins
    xlog = abstractproperty()   # whether to space bins logarithmically

    @abstractmethod
    def get_data(self):
        """
        Return array of bin counts
        """
        pass


class HistogramLayerArtist(MatplotlibLayerArtist, HistogramLayerBase):
    _property_set = MatplotlibLayerArtist._property_set + 'lo hi nbins xlog'.split()

    lo = ChangedTrigger(0)
    hi = ChangedTrigger(1)
    nbins = ChangedTrigger(10)
    xlog = ChangedTrigger(False)
    att = ChangedTrigger()

    def __init__(self, layer, axes):
        super(HistogramLayerArtist, self).__init__(layer, axes)
        self.ylog = False
        self.cumulative = False
        self.normed = False
        self.y = np.array([])
        self.x = np.array([])
        self._y = np.array([])

        self._scale_state = None

    def get_data(self):
        return self.x, self.y

    def clear(self):
        super(HistogramLayerArtist, self).clear()
        self.x = np.array([])
        self.y = np.array([])
        self._y = np.array([])

    def _calculate_histogram(self):
        """Recalculate the histogram, creating new patches"""
        self.clear()
        try:
            data = self.layer[self.att].ravel()
            if not np.isfinite(data).any():
                return False
        except IncompatibleAttribute as exc:
            self.disable_invalid_attributes(*exc.args)
            return False

        if data.size == 0:
            return

        if self.lo > np.nanmax(data) or self.hi < np.nanmin(data):
            return
        if self.xlog:
            data = np.log10(data)
            rng = [np.log10(self.lo), np.log10(self.hi)]
        else:
            rng = self.lo, self.hi
        nbinpatch = self._axes.hist(data,
                                    bins=int(self.nbins),
                                    range=rng)
        self._y, self.x, self.artists = nbinpatch
        return True

    def _scale_histogram(self):
        """Modify height of bins to match ylog, cumulative, and norm"""
        if self.x.size == 0:
            return

        y = self._y.astype(np.float)
        dx = self.x[1] - self.x[0]
        if self.normed:
            div = y.sum() * dx
            if div == 0:
                div = 1
            y /= div
        if self.cumulative:
            y = y.cumsum()
            y /= y.max()

        self.y = y
        bottom = 0 if not self.ylog else 1e-100

        for a, y in zip(self.artists, y):
            a.set_height(y)
            x, y = a.get_xy()
            a.set_xy((x, bottom))

    def _check_scale_histogram(self):
        """
        If needed, rescale histogram to match cumulative/log/normed state.
        """
        state = (self.normed, self.ylog, self.cumulative)
        if state == self._scale_state:
            return
        self._scale_state = state
        self._scale_histogram()

    def update(self, view=None):
        """Sync plot.

        The _change flag tracks whether the histogram needs to be
        recalculated. If not, the properties of the existing
        artists are updated
        """
        self._check_subset_state_changed()
        if self._changed:
            if not self._calculate_histogram():
                return
            self._changed = False
            self._scale_state = None
        self._check_scale_histogram()
        self._sync_style()

    def _sync_style(self):
        """Update visual properties"""
        style = self.layer.style
        for artist in self.artists:
            artist.set_facecolor(style.color)
            artist.set_alpha(style.alpha)
            artist.set_zorder(self.zorder)
            artist.set_visible(self.visible and self.enabled)
