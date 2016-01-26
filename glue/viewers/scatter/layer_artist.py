from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractproperty, abstractmethod

import numpy as np

from glue.external import six
from glue.core.subset import Subset
from glue.core.exceptions import IncompatibleAttribute
from glue.core.layer_artist import MatplotlibLayerArtist, ChangedTrigger


__all__ = ['ScatterLayerArtist']


@six.add_metaclass(ABCMeta)
class ScatterLayerBase(object):

    # which ComponentID to assign to X axis
    xatt = abstractproperty()

    # which ComponentID to assign to Y axis
    yatt = abstractproperty()

    @abstractmethod
    def get_data(self):
        """
        Returns
        -------
        array
            The scatterpoint data as an (N, 2) array
        """
        pass


class ScatterLayerArtist(MatplotlibLayerArtist, ScatterLayerBase):
    xatt = ChangedTrigger()
    yatt = ChangedTrigger()
    _property_set = MatplotlibLayerArtist._property_set + ['xatt', 'yatt']

    def __init__(self, layer, ax):
        super(ScatterLayerArtist, self).__init__(layer, ax)
        self.emphasis = None  # an optional SubsetState of emphasized points

    def _recalc(self):
        self.clear()
        assert len(self.artists) == 0

        try:
            x = self.layer[self.xatt].ravel()
            y = self.layer[self.yatt].ravel()
        except IncompatibleAttribute as exc:
            self.disable_invalid_attributes(*exc.args)
            return False

        self.artists = self._axes.plot(x, y)
        return True

    def update(self, view=None, transpose=False):
        self._check_subset_state_changed()

        if self._changed:  # erase and make a new artist
            if not self._recalc():  # no need to update style
                return
            self._changed = False

        has_emph = False
        if self.emphasis is not None:
            try:
                s = Subset(self.layer.data)
                s.subset_state = self.emphasis
                if hasattr(self.layer, 'subset_state'):
                    s.subset_state &= self.layer.subset_state
                x = s[self.xatt].ravel()
                y = s[self.yatt].ravel()
                self.artists.extend(self._axes.plot(x, y))
                has_emph = True
            except IncompatibleAttribute:
                pass

        self._sync_style()
        if has_emph:
            self.artists[-1].set_mec('green')
            self.artists[-1].set_mew(2)
            self.artists[-1].set_alpha(1)

    def get_data(self):
        try:
            return self.layer[self.xatt].ravel(), self.layer[self.yatt].ravel()
        except IncompatibleAttribute:
            return np.array([]), np.array([])
