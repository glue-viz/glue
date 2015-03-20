from __future__ import print_function, division

import sys
import os.path
import numpy as np

from ...external.qt.QtGui import (QAction,
                                  QToolButton, QToolBar, QIcon,
                                  QActionGroup, QWidget,
                                  QVBoxLayout, QColor, QImage, QPixmap)

from ...external.qt.QtCore import Qt, QSize

from ...qt.widgets.scatter_widget import ScatterWidget
from ...qt.widgets.data_viewer import DataViewer
from ...qt.widgets.mpl_widget import MplWidget, defer_draw
from ...qt.widget_properties import (ButtonProperty, FloatLineProperty,
                                 CurrentComboProperty,
                                 connect_bool_button, connect_float_edit)
from .client import ScatterGroupClient

from ...core import roi as roimod
from ...core.callback_property import add_callback

from ...qt.qtutil import get_icon, nonpartial, load_ui, cache_axes

from ..tools.pv_slicer import PVSlicerTool
from ..tools.spectrum_tool import SpectrumTool

from ...config import tool_registry

__all__ = ['ScatterGroupWidget']


class ScatterGroupWidget(ScatterWidget, DataViewer):

    LABEL = "Scatter Group Plot"
    _property_set = DataViewer._property_set + \
        'xlog ylog xflip yflip hidden xatt yatt xmin xmax ymin ymax gatt'.split()

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

    def __init__(self, session, parent=None):
        super(ScatterGroupWidget, self).__init__(session, parent)
        self.central_widget = MplWidget()
        self.option_widget = QWidget()

        self.setCentralWidget(self.central_widget)

        self.ui = load_ui('scattergroupwidget')
        self._tweak_geometry()

        self.client = ScatterGroupClient(self._data,
                                         self.central_widget.canvas.fig,
                                         artist_container=self._container)

        self._connect()
        self.unique_fields = set()
        tb = self.make_toolbar()
        cache_axes(self.client.axes, tb)
        self.statusBar().setSizeGripEnabled(False)
        self.setFocusPolicy(Qt.StrongFocus)

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