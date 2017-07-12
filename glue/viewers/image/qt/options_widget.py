from __future__ import absolute_import, division, print_function

import os

from qtpy import QtWidgets

from glue.core.data import Data
from glue.external.echo import delay_callback
from glue.external.echo.qt import autoconnect_callbacks_to_qt
from glue.utils.qt import load_ui
from glue.core.qt.data_combo_helper import ComponentIDComboHelper, ManualDataComboHelper
from glue.viewers.image.qt.slice_widget import MultiSliceWidgetHelper

__all__ = ['ImageOptionsWidget']


class ImageOptionsWidget(QtWidgets.QWidget):

    def __init__(self, viewer_state, session, parent=None):

        super(ImageOptionsWidget, self).__init__(parent=parent)

        self.ui = load_ui('options_widget.ui', self,
                          directory=os.path.dirname(__file__))

        viewer_state.add_callback('layers', self._update_combo_ref_data, priority=1500)
        viewer_state.add_callback('reference_data', self._update_combo_att)

        self.ui.combodata_aspect.addItem("Square Pixels", userData='equal')
        self.ui.combodata_aspect.addItem("Automatic", userData='auto')
        self.ui.combodata_aspect.setCurrentIndex(0)

        self.ui.combotext_color_mode.addItem("Colormaps")
        self.ui.combotext_color_mode.addItem("One color per layer")

        autoconnect_callbacks_to_qt(viewer_state, self.ui)

        self.ref_data_helper = ManualDataComboHelper(self.ui.combodata_reference_data,
                                                     session.data_collection)

        self.x_att_helper = ComponentIDComboHelper(self.ui.combodata_x_att_world,
                                                   session.data_collection,
                                                   numeric=False, categorical=False,
                                                   visible=False, world_coord=True,
                                                   default_index=-1)

        self.y_att_helper = ComponentIDComboHelper(self.ui.combodata_y_att_world,
                                                   session.data_collection,
                                                   numeric=False, categorical=False,
                                                   visible=False, world_coord=True,
                                                   default_index=-2)

        self.viewer_state = viewer_state

        self.slice_helper = MultiSliceWidgetHelper(viewer_state=self.viewer_state,
                                                   widget=self.ui.slice_tab)

    def _update_combo_ref_data(self, *args):
        datasets = []
        for layer in self.viewer_state.layers:
            if isinstance(layer.layer, Data):
                if layer.layer not in datasets:
                    datasets.append(layer.layer)
            else:
                if layer.layer.data not in datasets:
                    datasets.append(layer.layer.data)
        self.ref_data_helper.set_multiple_data(datasets)

    def _update_combo_att(self, *args):
        with delay_callback(self.viewer_state, 'x_att_world', 'y_att_world'):
            self.x_att_helper.set_multiple_data([self.viewer_state.reference_data])
            self.y_att_helper.set_multiple_data([self.viewer_state.reference_data])
