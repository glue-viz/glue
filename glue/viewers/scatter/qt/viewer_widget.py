from __future__ import absolute_import, division, print_function

import os

from qtpy.QtCore import Qt
from qtpy import QtWidgets
from glue import core
from glue.viewers.scatter.client import ScatterClient
from glue.viewers.common.qt.mpl_toolbar import MatplotlibViewerToolbar
from glue.viewers.common.qt.mouse_mode import (RectangleMode, CircleMode,
                                PolyMode, HRangeMode, VRangeMode)
from glue.utils.qt import load_ui
from glue.viewers.common.qt.data_viewer import DataViewer
from glue.viewers.common.qt.mpl_widget import MplWidget, defer_draw
from glue.viewers.scatter.qt.layer_style_widget import ScatterLayerStyleWidget
from glue.viewers.scatter.layer_artist import ScatterLayerArtist
from glue.utils import nonpartial, cache_axes
from glue.utils.qt.widget_properties import (ButtonProperty, FloatLineProperty,
                                             CurrentComboProperty,
                                             connect_bool_button, connect_float_edit)

__all__ = ['ScatterWidget']

WARN_SLOW = 1000000  # max number of points which render quickly


class ScatterWidget(DataViewer):

    """
    An interactive scatter plot.
    """

    LABEL = "Scatter Plot"
    _property_set = DataViewer._property_set + \
        'xlog ylog xflip yflip hidden xatt yatt xmin xmax ymin ymax'.split()

    xlog = ButtonProperty('ui.xLogCheckBox', 'log scaling on x axis?')
    ylog = ButtonProperty('ui.yLogCheckBox', 'log scaling on y axis?')
    xflip = ButtonProperty('ui.xFlipCheckBox', 'invert the x axis?')
    yflip = ButtonProperty('ui.yFlipCheckBox', 'invert the y axis?')
    xmin = FloatLineProperty('ui.xmin', 'Lower x limit of plot')
    xmax = FloatLineProperty('ui.xmax', 'Upper x limit of plot')
    ymin = FloatLineProperty('ui.ymin', 'Lower y limit of plot')
    ymax = FloatLineProperty('ui.ymax', 'Upper y limit of plot')
    hidden = ButtonProperty('ui.hidden_attributes', 'Show hidden attributes')
    xatt = CurrentComboProperty('ui.xAxisComboBox',
                                'Attribute to plot on x axis')
    yatt = CurrentComboProperty('ui.yAxisComboBox',
                                'Attribute to plot on y axis')

    _layer_style_widget_cls = {ScatterLayerArtist: ScatterLayerStyleWidget}

    _toolbar_cls = MatplotlibViewerToolbar
    tools = ['select:rectangle', 'select:xrange', 'select:yrange', 'select:circle', 'select:polygon']

    def __init__(self, session, parent=None):

        super(ScatterWidget, self).__init__(session, parent)

        self.central_widget = MplWidget()
        self.setCentralWidget(self.central_widget)

        self.option_widget = QtWidgets.QWidget()
        self.ui = load_ui('options_widget.ui', self.option_widget,
                          directory=os.path.dirname(__file__))

        self._tweak_geometry()

        self.client = ScatterClient(self._data,
                                    self.central_widget.canvas.fig,
                                    layer_artist_container=self._layer_artist_container)

        self._connect()
        self.unique_fields = set()
        self.statusBar().setSizeGripEnabled(False)
        self.setFocusPolicy(Qt.StrongFocus)

    def initialize_toolbar(self):
        super(ScatterWidget, self).initialize_toolbar()
        cache_axes(self.client.axes, self.toolbar)

    def _tweak_geometry(self):
        self.central_widget.resize(600, 400)
        self.resize(self.central_widget.size())

    def _connect(self):
        ui = self.ui
        cl = self.client

        connect_bool_button(cl, 'xlog', ui.xLogCheckBox)
        connect_bool_button(cl, 'ylog', ui.yLogCheckBox)
        connect_bool_button(cl, 'xflip', ui.xFlipCheckBox)
        connect_bool_button(cl, 'yflip', ui.yFlipCheckBox)

        ui.xAxisComboBox.currentIndexChanged.connect(self.update_xatt)
        ui.yAxisComboBox.currentIndexChanged.connect(self.update_yatt)
        ui.hidden_attributes.toggled.connect(lambda x: self._update_combos())
        ui.swapAxes.clicked.connect(nonpartial(self.swap_axes))
        ui.snapLimits.clicked.connect(cl.snap)

        connect_float_edit(cl, 'xmin', ui.xmin)
        connect_float_edit(cl, 'xmax', ui.xmax)
        connect_float_edit(cl, 'ymin', ui.ymin)
        connect_float_edit(cl, 'ymax', ui.ymax)

    @defer_draw
    def _update_combos(self):
        """ Update contents of combo boxes """

        # have to be careful here, since client and/or widget
        # are potentially out of sync

        layer_ids = []

        # show hidden attributes if needed
        if ((self.client.xatt and self.client.xatt.hidden) or
                (self.client.yatt and self.client.yatt.hidden)):
            self.hidden = True

        # determine which components to put in combos
        for l in self.client.data:
            if not self.client.is_layer_present(l):
                continue
            for lid in self.client.plottable_attributes(
                    l, show_hidden=self.hidden):
                if lid not in layer_ids:
                    layer_ids.append(lid)

        oldx = self.xatt
        oldy = self.yatt
        newx = self.client.xatt or oldx
        newy = self.client.yatt or oldy

        for combo, target in zip([self.ui.xAxisComboBox, self.ui.yAxisComboBox],
                                 [newx, newy]):
            combo.blockSignals(True)
            combo.clear()

            if not layer_ids:  # empty component list
                continue

            # populate
            for lid in layer_ids:
                combo.addItem(lid.label, userData=lid)

            idx = layer_ids.index(target) if target in layer_ids else 0
            combo.setCurrentIndex(idx)

            combo.blockSignals(False)

        # ensure client and widget synced
        self.client.xatt = self.xatt
        self.client.lyatt = self.yatt

    @defer_draw
    def add_data(self, data):
        """Add a new data set to the widget

        :returns: True if the addition was expected, False otherwise
        """
        if self.client.is_layer_present(data):
            return

        if data.size > WARN_SLOW and not self._confirm_large_data(data):
            return False

        first_layer = self.client.layer_count == 0

        self.client.add_data(data)
        self._update_combos()

        if first_layer:  # forces both x and y axes to be rescaled
            self.update_xatt(None)
            self.update_yatt(None)

            self.ui.xAxisComboBox.setCurrentIndex(0)
            if len(data.visible_components) > 1:
                self.ui.yAxisComboBox.setCurrentIndex(1)
            else:
                self.ui.yAxisComboBox.setCurrentIndex(0)

        self.update_window_title()
        return True

    @defer_draw
    def add_subset(self, subset):
        """Add a subset to the widget

        :returns: True if the addition was accepted, False otherwise
        """
        if self.client.is_layer_present(subset):
            return

        data = subset.data
        if data.size > WARN_SLOW and not self._confirm_large_data(data):
            return False

        first_layer = self.client.layer_count == 0

        self.client.add_layer(subset)
        self._update_combos()

        if first_layer:  # forces both x and y axes to be rescaled
            self.update_xatt(None)
            self.update_yatt(None)

            self.ui.xAxisComboBox.setCurrentIndex(0)
            if len(data.visible_components) > 1:
                self.ui.yAxisComboBox.setCurrentIndex(1)
            else:
                self.ui.yAxisComboBox.setCurrentIndex(0)

        self.update_window_title()
        return True

    def register_to_hub(self, hub):
        super(ScatterWidget, self).register_to_hub(hub)
        self.client.register_to_hub(hub)
        hub.subscribe(self, core.message.DataUpdateMessage,
                      nonpartial(self._sync_labels))
        hub.subscribe(self, core.message.ComponentsChangedMessage,
                      nonpartial(self._update_combos))
        hub.subscribe(self, core.message.ComponentReplacedMessage,
                      self._on_component_replace)

    def _on_component_replace(self, msg):
        # let client update its state first
        self.client._on_component_replace(msg)
        self._update_combos()

    def unregister(self, hub):
        super(ScatterWidget, self).unregister(hub)
        hub.unsubscribe_all(self.client)
        hub.unsubscribe_all(self)

    @defer_draw
    def swap_axes(self):
        xid = self.ui.xAxisComboBox.currentIndex()
        yid = self.ui.yAxisComboBox.currentIndex()
        xlog = self.ui.xLogCheckBox.isChecked()
        ylog = self.ui.yLogCheckBox.isChecked()
        xflip = self.ui.xFlipCheckBox.isChecked()
        yflip = self.ui.yFlipCheckBox.isChecked()

        self.ui.xAxisComboBox.setCurrentIndex(yid)
        self.ui.yAxisComboBox.setCurrentIndex(xid)
        self.ui.xLogCheckBox.setChecked(ylog)
        self.ui.yLogCheckBox.setChecked(xlog)
        self.ui.xFlipCheckBox.setChecked(yflip)
        self.ui.yFlipCheckBox.setChecked(xflip)

    @defer_draw
    def update_xatt(self, index):
        component_id = self.xatt
        self.client.xatt = component_id

    @defer_draw
    def update_yatt(self, index):
        component_id = self.yatt
        self.client.yatt = component_id

    @property
    def window_title(self):
        data = self.client.data
        label = ', '.join([d.label for d in data if
                           self.client.is_visible(d)])
        return label

    def _sync_labels(self):
        self.update_window_title()

    def options_widget(self):
        return self.option_widget

    @defer_draw
    def restore_layers(self, rec, context):
        self.client.restore_layers(rec, context)
        self._update_combos()
        # manually force client attributes to sync
        self.update_xatt(None)
        self.update_yatt(None)
