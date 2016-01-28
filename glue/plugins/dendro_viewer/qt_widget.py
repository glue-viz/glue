from __future__ import absolute_import, division, print_function

import os

from glue.external.qt import QtGui
from glue import core
from glue.plugins.dendro_viewer.client import DendroClient
from glue.viewers.common.qt.toolbar import GlueToolbar
from glue.qt.mouse_mode import PickMode
from glue.qt.qtutil import load_ui
from glue.utils.qt.widget_properties import (ButtonProperty, CurrentComboProperty,
                                       connect_bool_button, connect_current_combo)
from glue.viewers.common.qt.data_viewer import DataViewer
from glue.viewers.common.qt.mpl_widget import MplWidget, defer_draw
from glue.utils import nonpartial


class DendroWidget(DataViewer):

    """
    An interactive dendrogram display
    """

    LABEL = 'Dendrogram'

    _property_set = DataViewer._property_set + \
        'ylog height parent order'.split()

    ylog = ButtonProperty('ui.ylog', 'log scaling on y axis?')
    height = CurrentComboProperty('ui.heightCombo', 'height attribute')
    parent = CurrentComboProperty('ui.parentCombo', 'parent attribute')
    order = CurrentComboProperty('ui.orderCombo', 'layout sorter attribute')

    def __init__(self, session, parent=None):
        super(DendroWidget, self).__init__(session, parent)

        self.central_widget = MplWidget()
        self.option_widget = QtGui.QWidget()
        self.setCentralWidget(self.central_widget)

        self.ui = load_ui('qt_widget.ui', self.option_widget,
                          directory=os.path.dirname(__file__))
        self.client = DendroClient(self._data,
                                   self.central_widget.canvas.fig,
                                   layer_artist_container=self._layer_artist_container)

        self._connect()

        self.make_toolbar()
        self.statusBar().setSizeGripEnabled(False)

    @staticmethod
    def _get_default_tools():
        return []

    def _connect(self):
        ui = self.ui
        cl = self.client

        connect_bool_button(cl, 'ylog', ui.ylog)
        connect_current_combo(cl, 'parent_attr', ui.parentCombo)
        connect_current_combo(cl, 'height_attr', ui.heightCombo)
        connect_current_combo(cl, 'order_attr', ui.orderCombo)

    def make_toolbar(self):
        result = GlueToolbar(self.central_widget.canvas, self,
                             name='Dendrogram')
        for mode in self._mouse_modes():
            result.add_mode(mode)
        self.addToolBar(result)
        return result

    def _mouse_modes(self):
        axes = self.client.axes

        def apply_mode(mode):
            self.client.apply_roi(mode.roi())

        def on_move(mode):
            if mode._drag:
                self.client.apply_roi(mode.roi())

        return [PickMode(axes,
                         move_callback=on_move,
                         roi_callback=apply_mode)]

    def _update_combos(self, data=None):
        data = data or self.client.display_data
        if data is None:
            return

        for combo in [self.ui.heightCombo,
                      self.ui.parentCombo,
                      self.ui.orderCombo]:
            combo.blockSignals(True)
            ids = []
            idx = combo.currentIndex()
            old = combo.itemData(idx) if idx > 0 else None
            combo.clear()
            for cid in data.components:
                if cid.hidden and cid is not data.pixel_component_ids[0]:
                    continue
                combo.addItem(cid.label, userData=cid)
                ids.append(cid)
            try:
                combo.setCurrentIndex(ids.index(old))
            except ValueError:
                combo.setCurrentIndex(0)

            combo.blockSignals(False)

    def add_data(self, data):
        """Add a new data set to the widget

        :returns: True if the addition was expected, False otherwise
        """
        if data in self.client:
            return

        self._update_combos(data)
        self.client.add_layer(data)
        return True

    def add_subset(self, subset):
        """Add a subset to the widget

        :returns: True if the addition was accepted, False otherwise
        """
        self.add_data(subset.data)
        if subset.data in self.client:
            self.client.add_layer(subset)
            return True

    def register_to_hub(self, hub):
        super(DendroWidget, self).register_to_hub(hub)
        self.client.register_to_hub(hub)
        hub.subscribe(self, core.message.ComponentsChangedMessage,
                      nonpartial(self._update_combos()))

    def unregister(self, hub):
        super(DendroWidget, self).unregister(hub)
        hub.unsubscribe_all(self.client)
        hub.unsubscribe_all(self)

    def options_widget(self):
        return self.option_widget

    @defer_draw
    def restore_layers(self, rec, context):

        from glue.core.callback_property import delay_callback

        with delay_callback(self.client, 'height_attr',
                                         'parent_attr',
                                         'order_attr'):
            self.client.restore_layers(rec, context)
            self._update_combos()
