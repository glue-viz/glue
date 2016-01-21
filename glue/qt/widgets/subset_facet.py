from __future__ import absolute_import, division, print_function

from glue.external.qt import QtGui
import numpy as np
from matplotlib import cm


from glue.qt.qtutil import pretty_number, cmap2pixmap, load_ui
from glue.core.util import colorize_subsets, facet_subsets, Pointer
from glue.qt.widget_properties import (ButtonProperty, FloatLineProperty,
                                 ValueProperty)


class SubsetFacet(object):
    log = ButtonProperty('ui.log')
    vmin = FloatLineProperty('ui.min')
    vmax = FloatLineProperty('ui.max')
    steps = ValueProperty('ui.num')
    data = Pointer('ui.component_selector.data')
    component = Pointer('ui.component_selector.component')

    def __init__(self, collect, default=None, parent=None):
        """Create a new dialog for subset faceting

        :param collect: The :class:`~glue.core.data_collection.DataCollection` to use
        :param default: The default dataset in the collection (optional)
        """
        self.ui = load_ui('subset_facet', None)
        self.ui.setWindowTitle("Subset Facet")
        self._collect = collect

        self.ui.component_selector.setup(self._collect)
        if default is not None:
            self.ui.component_selector.data = default

        val = QtGui.QDoubleValidator(-1e100, 1e100, 4, None)
        self.ui.component_selector.component_changed.connect(self._set_limits)

        combo = self.ui.color_scale
        for cmap in [cm.cool, cm.RdYlBu, cm.RdYlGn, cm.RdBu, cm.Purples]:
            combo.addItem(QtGui.QIcon(cmap2pixmap(cmap)), cmap.name, cmap)

    def _set_limits(self):
        data = self.ui.component_selector.data
        cid = self.ui.component_selector.component

        vals = data[cid]

        wmin = self.ui.min
        wmax = self.ui.max

        wmin.setText(pretty_number(np.nanmin(vals)))
        wmax.setText(pretty_number(np.nanmax(vals)))

    @property
    def cmap(self):
        combo = self.ui.color_scale
        index = combo.currentIndex()
        return combo.itemData(index)

    def _apply(self):
        try:
            lo, hi = self.vmin, self.vmax
        except ValueError:
            return  # limits not set. Abort
        if not np.isfinite(lo) or not np.isfinite(hi):
            return

        subsets = facet_subsets(self._collect, self.component, lo=lo, hi=hi,
                                steps=self.steps, log=self.log)
        colorize_subsets(subsets, self.cmap)

    def exec_(self):
        return self.ui.exec_()

    @classmethod
    def facet(cls, collect, default=None, parent=None):
        """Class method to create facted subsets
        The arguments are the same as __init__
        """
        self = cls(collect, parent=parent, default=default)
        value = self.exec_()

        if value == QtGui.QDialog.Accepted:
            self._apply()
