from __future__ import absolute_import, division, print_function

import os
from functools import partial

from qtpy.QtCore import Qt
from qtpy import QtGui, QtWidgets
from glue.core import message as msg
from glue.viewers.histogram.client import HistogramClient
from glue.viewers.common.qt.mpl_toolbar import MatplotlibViewerToolbar
from glue.utils.qt import load_ui
from glue.utils.qt.widget_properties import (connect_int_spin, ButtonProperty,
                                             FloatLineProperty, connect_float_edit,
                                             ValueProperty, connect_bool_button)
from glue.viewers.common.qt.data_viewer import DataViewer
from glue.viewers.common.qt.mpl_widget import MplWidget, defer_draw
from glue.viewers.histogram.qt.layer_style_widget import HistogramLayerStyleWidget
from glue.viewers.histogram.layer_artist import HistogramLayerArtist

__all__ = ['HistogramWidget']


WARN_SLOW = 10000000


def _hash(x):
    return str(id(x))


class HistogramWidget(DataViewer):
    LABEL = "Histogram"
    _property_set = DataViewer._property_set + \
        'component xlog ylog normed cumulative autoscale xmin xmax nbins'.split(
        )

    xmin = FloatLineProperty('ui.xmin', 'Minimum value')
    xmax = FloatLineProperty('ui.xmax', 'Maximum value')
    normed = ButtonProperty('ui.normalized_box', 'Normalized?')
    autoscale = ButtonProperty('ui.autoscale_box',
                               'Autoscale view to histogram?')
    cumulative = ButtonProperty('ui.cumulative_box', 'Cumulative?')
    nbins = ValueProperty('ui.binSpinBox', 'Number of bins')
    xlog = ButtonProperty('ui.xlog_box', 'Log-scale the x axis?')
    ylog = ButtonProperty('ui.ylog_box', 'Log-scale the y axis?')

    _layer_style_widget_cls = {HistogramLayerArtist: HistogramLayerStyleWidget}

    _toolbar_cls = MatplotlibViewerToolbar
    tools = ['select:xrange']

    def __init__(self, session, parent=None):
        super(HistogramWidget, self).__init__(session, parent)

        self.central_widget = MplWidget()
        self.setCentralWidget(self.central_widget)
        self.option_widget = QtWidgets.QWidget()
        self.ui = load_ui('options_widget.ui', self.option_widget,
                          directory=os.path.dirname(__file__))
        self._tweak_geometry()
        self.client = HistogramClient(self._data,
                                      self.central_widget.canvas.fig,
                                      layer_artist_container=self._layer_artist_container)
        self._init_limits()
        self._connect()
        # maps _hash(componentID) -> componentID
        self._component_hashes = {}

    def _init_limits(self):
        validator = QtGui.QDoubleValidator(None)
        validator.setDecimals(7)
        self.ui.xmin.setValidator(validator)
        self.ui.xmax.setValidator(validator)
        lo, hi = self.client.xlimits
        self.ui.xmin.setText(str(lo))
        self.ui.xmax.setText(str(hi))

    def _tweak_geometry(self):
        self.central_widget.resize(600, 400)
        self.resize(self.central_widget.size())

    def _connect(self):

        ui = self.ui
        cl = self.client

        ui.attributeCombo.currentIndexChanged.connect(self._set_attribute_from_combo)

        ui.normalized_box.toggled.connect(partial(setattr, cl, 'normed'))
        ui.autoscale_box.toggled.connect(partial(setattr, cl, 'autoscale'))
        ui.cumulative_box.toggled.connect(partial(setattr, cl, 'cumulative'))

        connect_int_spin(cl, 'nbins', ui.binSpinBox)
        connect_float_edit(cl, 'xmin', ui.xmin)
        connect_float_edit(cl, 'xmax', ui.xmax)
        connect_bool_button(cl, 'xlog', ui.xlog_box)
        connect_bool_button(cl, 'ylog', ui.ylog_box)

    @defer_draw
    def _update_attributes(self):
        """Repopulate the combo box that selects the quantity to plot"""
        combo = self.ui.attributeCombo
        component = self.component
        new = self.client.component or component

        combo.blockSignals(True)
        combo.clear()

        # implementation note:
        # PySide doesn't robustly store python objects with setData
        # use _hash(x) instead
        model = QtGui.QStandardItemModel()
        data_ids = set(_hash(d) for d in self._data)
        self._component_hashes = dict((_hash(c), c) for d in self._data
                                      for c in d.components)

        found = False
        for d in self._data:
            if d not in self._layer_artist_container:
                continue
            item = QtGui.QStandardItem(d.label)
            item.setData(_hash(d), role=Qt.UserRole)
            assert item.data(Qt.UserRole) == _hash(d)
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            model.appendRow(item)
            for c in d.visible_components:
                if (not d.get_component(c).categorical and
                        not d.get_component(c).numeric):
                    continue
                if c is new:
                    found = True
                item = QtGui.QStandardItem(c.label)
                item.setData(_hash(c), role=Qt.UserRole)
                model.appendRow(item)
        combo.setModel(model)

        # separators below data items
        for i in range(combo.count()):
            if combo.itemData(i) in data_ids:
                combo.insertSeparator(i + 1)

        combo.blockSignals(False)

        if found:
            self.component = new
        else:
            combo.setCurrentIndex(2)  # skip first data + separator
        self._set_attribute_from_combo()

    @property
    def component(self):
        combo = self.ui.attributeCombo
        index = combo.currentIndex()
        return self._component_hashes.get(combo.itemData(index), None)

    @component.setter
    def component(self, component):
        combo = self.ui.attributeCombo
        if combo.count() == 0:  # cold start problem, when restoring
            self._update_attributes()

        # combo.findData doesn't seem to work robustly
        for i in range(combo.count()):
            data = combo.itemData(i)
            if data == _hash(component):
                combo.setCurrentIndex(i)
                return
        raise IndexError("Component not present: %s" % component)

    @defer_draw
    def _set_attribute_from_combo(self, *args):
        if self.component is not None:
            for d in self._data:
                try:
                    component = d.get_component(self.component)
                except:
                    continue
                else:
                    break
            if component.categorical:
                if self.ui.xlog_box.isEnabled():
                    self.ui.xlog_box.setEnabled(False)
                    self.xlog = False
            else:
                if not self.ui.xlog_box.isEnabled():
                    self.ui.xlog_box.setEnabled(True)
        self.client.set_component(self.component)
        self.update_window_title()

    @defer_draw
    def add_data(self, data):
        """ Add data item to combo box.
        If first addition, also update attributes """

        if self.data_present(data):
            return True

        if data.size > WARN_SLOW and not self._confirm_large_data(data):
            return False

        self.client.add_layer(data)
        self._update_attributes()

        return True

    def add_subset(self, subset):
        pass

    def _remove_data(self, data):
        """ Remove data item from the combo box """
        pass

    def data_present(self, data):
        return data in self._layer_artist_container

    def register_to_hub(self, hub):
        super(HistogramWidget, self).register_to_hub(hub)
        self.client.register_to_hub(hub)

        hub.subscribe(self,
                      msg.DataCollectionDeleteMessage,
                      handler=lambda x: self._remove_data(x.data))

        hub.subscribe(self,
                      msg.DataUpdateMessage,
                      handler=lambda *args: self._update_labels())

        hub.subscribe(self,
                      msg.ComponentsChangedMessage,
                      handler=lambda x: self._update_attributes())

    def unregister(self, hub):
        super(HistogramWidget, self).unregister(hub)
        self.client.unregister(hub)
        hub.unsubscribe_all(self)

    @property
    def window_title(self):
        c = self.client.component
        if c is not None:
            label = str(c.label)
        else:
            label = 'Histogram'
        return label

    def _update_labels(self):
        self.update_window_title()
        self._update_attributes()

    def __str__(self):
        return "Histogram Widget"

    def options_widget(self):
        return self.option_widget
