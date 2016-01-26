from __future__ import absolute_import, division, print_function

import numpy as np

from glue.core.exceptions import IncompatibleAttribute
from glue.core.subset import Subset
from glue.core.layer_artist import MatplotlibLayerArtist, ChangedTrigger


class DendroLayerArtist(MatplotlibLayerArtist):
    # X vertices of structure i are in layout[0][3*i: 3*i+3]
    layout = ChangedTrigger()

    def __init__(self, layer, ax):
        super(DendroLayerArtist, self).__init__(layer, ax)

    def _recalc(self):
        self.clear()
        assert len(self.artists) == 0
        if self.layout is None:
            return

        # layout[0] is [x0, x0, x[parent0], nan, ...]
        # layout[1] is [y0, y[parent0], y[parent0], nan, ...]
        ids = 3 * np.arange(self.layer.data.size)

        try:
            if isinstance(self.layer, Subset):
                ids = ids[self.layer.to_mask()]

            x, y = self.layout
            blank = np.zeros(ids.size) * np.nan
            x = np.column_stack([x[ids], x[ids + 1],
                                 x[ids + 2], blank]).ravel()
            y = np.column_stack([y[ids], y[ids + 1],
                                 y[ids + 2], blank]).ravel()
        except IncompatibleAttribute as exc:
            self.disable_invalid_attributes(*exc.args)
            return False

        self.artists = self._axes.plot(x, y, '--')
        return True

    def update(self, view=None):
        self._check_subset_state_changed()

        if self._changed:  # erase and make a new artist
            if not self._recalc():  # no need to update style
                return
            self._changed = False

        self._sync_style()

    def _sync_style(self):
        super(DendroLayerArtist, self)._sync_style()
        style = self.layer.style
        lw = 4 if isinstance(self.layer, Subset) else 2
        for artist in self.artists:
            artist.set_linestyle('-')
            artist.set_marker(None)
            artist.set_color(style.color)
            artist.set_linewidth(lw)
