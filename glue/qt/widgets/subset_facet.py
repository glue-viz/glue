from ...external.qt.QtGui import (QDialog, QDoubleValidator, QIcon)
import numpy as np
from matplotlib import cm


from ..qtutil import pretty_number, cmap2pixmap, load_ui
from ...core.util import colorize_subsets, facet_subsets
from ..widget_properties import ButtonProperty


class SubsetFacet(QDialog):
    log = ButtonProperty('ui.log')

    def __init__(self, collect, default=None, parent=None):
        """Create a new dialog for subset faceting

        :param collect: The :class:`~glue.core.DataCollection` to use
        :param default: The default dataset in the collection (optional)
        """
        super(SubsetFacet, self).__init__(parent)
        self.setWindowTitle("Subset Facet")
        self.ui = load_ui('subset_facet', self)
        self._collect = collect

        self.ui.component_selector.setup(self._collect)
        if default is not None:
            self.ui.component_selector.data = default

        val = QDoubleValidator(-1e100, 1e100, 4, None)
        self.ui.component_selector.component_changed.connect(self._set_limits)

        combo = self.ui.color_scale
        for cmap in [cm.cool, cm.RdYlBu, cm.RdYlGn, cm.RdBu, cm.Purples]:
            combo.addItem(QIcon(cmap2pixmap(cmap)), cmap.name, cmap)

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
        lo, hi = self.ui.min.text(), self.ui.max.text()
        try:
            lo, hi = float(lo), float(hi)
        except ValueError:
            return  # limits not set. Abort
        if not np.isfinite(lo) or not np.isfinite(hi):
            return

        steps = self.ui.num.value()

        data = self.ui.component_selector.data
        cid = self.ui.component_selector.component

        subsets = facet_subsets(data, cid, lo=lo, hi=hi,
                                steps=steps, log=self.log)
        colorize_subsets(subsets, self.cmap)

    @classmethod
    def facet(cls, collect, default=None, parent=None):
        """Class method to create facted subsets
        The arguments are the same as __init__
        """
        self = cls(collect, parent=parent, default=default)
        value = self.exec_()

        if value == QDialog.Accepted:
            self._apply()


def main():
    from glue.core import Data, DataCollection
    from glue.qt import get_qapp
    app = get_qapp()

    d = Data(label='d1', x=[1, 2, 3], y=[2, 3, 4])
    d2 = Data(label='d2', z=[1, 2, 3], w=[2, 3, 4])

    dc = DataCollection([d, d2])
    SubsetFacet.facet(dc)

    print 'd1 subsets'
    for s in d.subsets:
        print s.label, s.subset_state, s.style.color

    print 'd2 subsets'
    for s in d2.subsets:
        print s.label, s.subset_state, s.style.color

    del app

if __name__ == "__main__":
    main()
