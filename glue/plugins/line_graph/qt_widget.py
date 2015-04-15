from __future__ import print_function, division
import os.path

from ...qt.widgets.scatter_widget import ScatterWidget
from ...qt.widgets.mpl_widget import defer_draw
from ...qt.widget_properties import CurrentComboProperty

from ...qt.qtutil import load_ui
from .client import LineClient

__all__ = ['LineWidget']


class LineWidget(ScatterWidget):

    LABEL = "Line Graph"
    _property_set = ScatterWidget._property_set + ['gatt']
    gatt = CurrentComboProperty('ui.groupComboBox',
                                'Attribute to group information by')

    def _load_ui(self):
        self.ui = load_ui(os.path.join(
            os.path.dirname(__file__), 'linewidget.ui'),
            self.option_widget)

    def _setup_client(self):
        self.client = LineClient(self._data,
                                 self.central_widget.canvas.fig,
                                 artist_container=self._container)

    def _connect(self):
        ui = self.ui
        ui.groupComboBox.currentIndexChanged.connect(self.update_gatt)
        super(LineWidget, self)._connect()

    @defer_draw
    def _update_combos(self):
        super(LineWidget, self)._update_combos()
        glayer_ids = []
        if (self.client.gatt and self.client.gatt.hidden):
            self.hidden = True
        for l in self.client.data:
            if not self.client.is_layer_present(l):
                continue
            for glid in self.client.groupable_attributes(
                    l, show_hidden=self.hidden):
                if glid not in glayer_ids:
                    glayer_ids.append(glid)

        oldg = self.gatt
        newg = self.client.gatt or oldg
        for combo, target in zip([self.ui.groupComboBox], [newg]):
            combo.blockSignals(True)
            combo.clear()
            if not glayer_ids:
                continue
            for glid in glayer_ids:
                combo.addItem(glid.label, userData=glid)
            idx = glayer_ids.index(target) if target in glayer_ids else 0
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)
        self.client.gatt = self.gatt

    @defer_draw
    def update_gatt(self, index):
        component_id = self.gatt
        self.client.gatt = component_id

    @defer_draw
    def restore_layers(self, rec, context):
        super(LineWidget, self).restore_layers(rec, context)
        self.update_gatt(None)