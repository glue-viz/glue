from __future__ import print_function, division
import os.path

from ...qt.widgets.scatter_widget import ScatterWidget
from ...qt.widgets.mpl_widget import defer_draw
from ...qt.widget_properties import (ButtonProperty, FloatLineProperty,
                                 CurrentComboProperty,
                                 connect_bool_button, connect_float_edit)

from ...qt.qtutil import nonpartial, load_ui
from .client import ScatterGroupClient

__all__ = ['ScatterGroupWidget']


class ScatterGroupWidget(ScatterWidget):

    LABEL = "Scatter Group Plot"
    _property_set = ScatterWidget._property_set + ['gatt']

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
    gatt = CurrentComboProperty('ui.groupComboBox',
                                'Attribute to group time series')

    def _load_ui(self):
        self.ui = load_ui(os.path.join(os.path.dirname(__file__), 'scattergroupwidget.ui'), self.option_widget)

    def _setup_client(self):
        self.client = ScatterGroupClient(self._data,
                                         self.central_widget.canvas.fig,
                                         artist_container=self._container)

    def _connect(self):
         ui = self.ui
         cl = self.client

         connect_bool_button(cl, 'xlog', ui.xLogCheckBox)
         connect_bool_button(cl, 'ylog', ui.yLogCheckBox)
         connect_bool_button(cl, 'xflip', ui.xFlipCheckBox)
         connect_bool_button(cl, 'yflip', ui.yFlipCheckBox)

         ui.xAxisComboBox.currentIndexChanged.connect(self.update_xatt)
         ui.yAxisComboBox.currentIndexChanged.connect(self.update_yatt)
         ui.groupComboBox.currentIndexChanged.connect(self.update_gatt)
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
        glayer_ids = []

        # show hidden attributes if needed
        if ((self.client.xatt and self.client.xatt.hidden) or
                (self.client.yatt and self.client.yatt.hidden) or
                (self.client.gatt and self.client.gatt.hidden)):
            self.hidden = True

        # determine which components to put in combos
        for l in self.client.data:
            if not self.client.is_layer_present(l):
                continue
            for lid in self.client.plottable_attributes(
                    l, show_hidden=self.hidden):
                if lid not in layer_ids:
                    layer_ids.append(lid)
                for glid in self.client.groupable_attributes(
                        l, show_hidden=self.hidden):
                    if glid not in glayer_ids:
                        glayer_ids.append(glid)

        oldx = self.xatt
        oldy = self.yatt
        oldg = self.gatt
        newx = self.client.xatt or oldx
        newy = self.client.yatt or oldy
        newg = self.client.gatt or oldg

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

        for combo, target in zip([self.ui.groupComboBox], [newg]):

            combo.blockSignals(True)
            combo.clear()

            if not glayer_ids:  # empty component list
                continue

            # populate
            for glid in glayer_ids:
                combo.addItem(glid.label, userData=glid)

            idx = glayer_ids.index(target) if target in glayer_ids else 0
            combo.setCurrentIndex(idx)

            combo.blockSignals(False)

        # ensure client and widget synced
        self.client.xatt = self.xatt
        self.client.lyatt = self.yatt
        self.client.yatt = self.yatt
        self.client.gatt = self.gatt

    @defer_draw
    def update_gatt(self, index):
        component_id = self.gatt
        self.client.gatt = component_id

    @defer_draw
    def restore_layers(self, rec, context):
        self.client.restore_layers(rec, context)
        self._update_combos()
        # manually force client attributes to sync
        self.update_xatt(None)
        self.update_yatt(None)
        self.update_gatt(None)