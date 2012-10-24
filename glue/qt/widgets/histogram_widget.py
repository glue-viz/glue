from functools import partial

import numpy as np

from PyQt4 import QtGui

from ...core import message as msg
from ...clients.histogram_client import HistogramClient
from ..ui.histogramwidget import Ui_HistogramWidget
from ..glue_toolbar import GlueToolbar
from ..mouse_mode import RectangleMode
from .data_viewer import DataViewer
from .mpl_widget import MplWidget
from ..qtutil import pretty_number

WARN_SLOW = 10000000


class HistogramWidget(DataViewer):
    LABEL = "Histogram"

    def __init__(self, data, parent=None):
        super(HistogramWidget, self).__init__(self, parent)

        self.central_widget = MplWidget()
        self.setCentralWidget(self.central_widget)
        self.option_widget = QtGui.QWidget()
        self.ui = Ui_HistogramWidget()
        self.ui.setupUi(self.option_widget)
        self._tweak_geometry()
        self.client = HistogramClient(data,
                                      self.central_widget.canvas.fig,
                                      artist_container=self._container)

        validator = QtGui.QDoubleValidator(None)
        validator.setDecimals(7)
        self.ui.xmin.setValidator(validator)
        self.ui.xmax.setValidator(validator)
        lo, hi = self.client.xlimits
        self.ui.xmin.setText(str(lo))
        self.ui.xmax.setText(str(hi))
        self.make_toolbar()
        self._connect()
        self._data = data

    def _tweak_geometry(self):
        self.central_widget.resize(600, 400)
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
        ui.xmin.editingFinished.connect(self._set_limits)
        ui.xmax.editingFinished.connect(self._set_limits)

    def _set_limits(self):
        lo = float(self.ui.xmin.text())
        hi = float(self.ui.xmax.text())
        self.client.xlimits = lo, hi

    def _update_minmax_labels(self):
        lo, hi = pretty_number(self.client.xlimits)
        self.ui.xmin.setText(lo)
        self.ui.xmax.setText(hi)

    def make_toolbar(self):
        result = GlueToolbar(self.central_widget.canvas, self,
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
        component = self.component

        combo.clear()

        data = [a.layer.data for a in self._container]

        try:
            combo.currentIndexChanged.disconnect()
        except TypeError:
            pass

        for d in data:
            for c in d.visible_components:
                if not np.can_cast(d.dtype(c), np.float):
                    continue
                combo.addItem("%s (%s)" % (c.label, d.label), userData=c)

        combo.currentIndexChanged.connect(self._set_attribute_from_combo)
        combo.currentIndexChanged.connect(self._update_minmax_labels)
        if component is not None:
            self.component = component
        else:
            combo.setCurrentIndex(0)
        self._set_attribute_from_combo()

    @property
    def component(self):
        combo = self.ui.attributeCombo
        index = combo.currentIndex()
        return combo.itemData(index)

    @component.setter
    def component(self, component):
        combo = self.ui.attributeCombo
        #combo.findData doesn't seem to work in PyQt4
        for i in range(combo.count()):
            data = combo.itemData(i)
            if data is component:
                combo.setCurrentIndex(i)
                return
        raise IndexError("Component not present: %s" % component)

    def _set_attribute_from_combo(self):
        self.client.set_component(self.component)
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
        return data in self._container

    def register_to_hub(self, hub):
        super(HistogramWidget, self).register_to_hub(hub)
        self.client.register_to_hub(hub)

        hub.subscribe(self,
                      msg.DataCollectionDeleteMessage,
                      handler=lambda x: self._remove_data(x.data))

        hub.subscribe(self,
                      msg.DataUpdateMessage,
                      handler=lambda *args: self._update_labels())

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

    def _update_labels(self):
        self._update_window_title()
        self._update_attributes()

    def __str__(self):
        return "Histogram Widget"

    def options_widget(self):
        return self.option_widget
