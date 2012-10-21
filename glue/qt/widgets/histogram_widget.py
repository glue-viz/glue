from functools import partial

import numpy as np

from PyQt4 import QtGui

from ...core import message as msg
from ...clients.histogram_client import HistogramClient
from ..ui.histogramwidget import Ui_HistogramWidget
from ..glue_toolbar import GlueToolbar
from ..mouse_mode import RectangleMode
from .data_viewer import DataViewer
from ..layer_artist_model import QtLayerArtistContainer

WARN_SLOW = 10000000


class HistogramWidget(DataViewer):
    LABEL = "Histogram"

    def __init__(self, data, parent=None):
        super(HistogramWidget, self).__init__(self, parent)

        self.central_widget = QtGui.QWidget()
        self.setCentralWidget(self.central_widget)
        self.ui = Ui_HistogramWidget()
        self.ui.setupUi(self.central_widget)
        container = QtLayerArtistContainer()
        self._artist_container = container

        self.client = HistogramClient(data,
                                      self.ui.mplWidget.canvas.fig,
                                      artist_container=container)
        self.ui.artist_view.setModel(container.model)
        self.ui.xmin.setValidator(QtGui.QDoubleValidator())
        self.ui.xmax.setValidator(QtGui.QDoubleValidator())
        lo, hi = self.client.xlimits
        self.ui.xmin.setText(str(lo))
        self.ui.xmax.setText(str(hi))
        self.make_toolbar()
        self._connect()
        self._data = data
        self._tweak_geometry()

    def _tweak_geometry(self):
        self.central_widget.resize(600, 400)
        self.ui.splitter.setSizes([350, 120])
        self.ui.main_splitter.setSizes([300, 100])
        self.resize(self.central_widget.size())

    def _connect(self):
        ui = self.ui
        cl = self.client
        ui.attributeCombo.currentIndexChanged.connect(
            self._set_attribute_from_combo)
        ui.attributeCombo.currentIndexChanged.connect(
            self._update_minmax_labels)
        ui.binSpinBox.valueChanged.connect(partial(setattr, cl, 'nbins'))
        ui.normalized_box.toggled.connect(partial(setattr, cl, 'normed'))
        ui.autoscale_box.toggled.connect(partial(setattr, cl, 'autoscale'))
        ui.cumulative_box.toggled.connect(partial(setattr, cl, 'cumulative'))
        ui.xlog_box.toggled.connect(partial(setattr, cl, 'xlog'))
        ui.ylog_box.toggled.connect(partial(setattr, cl, 'ylog'))
        ui.xmin.returnPressed.connect(self._set_limits)
        ui.xmax.returnPressed.connect(self._set_limits)

    def _set_limits(self):
        lo = float(self.ui.xmin.text())
        hi = float(self.ui.xmax.text())
        self.client.xlimits = lo, hi

    def _update_minmax_labels(self):
        lo, hi = self.client.xlimits
        self.ui.xmin.setText(str(lo))
        self.ui.xmax.setText(str(hi))

    def make_toolbar(self):
        result = GlueToolbar(self.ui.mplWidget.canvas, self,
                             name='Histogram')
        for mode in self._mouse_modes():
            result.add_mode(mode)
        self.addToolBar(result)
        return result

    def _mouse_modes(self):
        axes = self.client.axes
        rect = RectangleMode(axes, release_callback=self.apply_roi)
        return [rect]

    def apply_roi(self, mode):
        roi = mode.roi()
        self.client.apply_roi(roi)

    def _update_attributes(self):
        combo = self.ui.attributeCombo
        combo.clear()

        data = [a.layer.data for a in self._artist_container]
        comps = set(c for d in data for c in d.visible_components if
                    np.can_cast(d[c].dtype, np.float))

        try:
            combo.currentIndexChanged.disconnect()
        except TypeError:
            pass

        for comp in comps:
            combo.addItem(comp.label, userData=comp)

        combo.currentIndexChanged.connect(self._set_attribute_from_combo)
        combo.currentIndexChanged.connect(self._update_minmax_labels)
        combo.setCurrentIndex(0)
        self._set_attribute_from_combo()

    def _set_attribute_from_combo(self):
        combo = self.ui.attributeCombo
        index = combo.currentIndex()
        attribute = combo.itemData(index)
        self.client.set_component(attribute)
        self._update_window_title()

    def add_data(self, data):
        """ Add data item to combo box.
        If first addition, also update attributes """

        if self.data_present(data):
            return True

        if data.size > WARN_SLOW and not self._confirm_large_data(data):
            return False

        self.client.add_layer(data)
        self._update_attributes()
        self._update_minmax_labels()
        return True

    def add_subset(self, subset):
        pass

    def _remove_data(self, data):
        """ Remove data item from the combo box """
        pass

    def data_present(self, data):
        return data in self._artist_container

    def register_to_hub(self, hub):
        super(HistogramWidget, self).register_to_hub(hub)
        self.client.register_to_hub(hub)

        hub.subscribe(self,
                      msg.DataCollectionDeleteMessage,
                      handler=lambda x: self._remove_data(x.data))

        hub.subscribe(self,
                      msg.DataUpdateMessage,
                      handler=lambda x: self._sync_data_labels())

    def unregister(self, hub):
        self.client.unregister(hub)
        hub.unsubscribe_all(self)

    def _update_window_title(self):
        c = self.client.component
        if c is not None:
            label = str(c.label)
        else:
            label = 'Histogram'
        self.setWindowTitle(label)

    def _sync_data_labels(self):
        self._update_window_title()
        self._update_attributes()
