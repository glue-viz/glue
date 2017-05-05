from __future__ import absolute_import, division, print_function

import os

from qtpy import QtWidgets

from glue.external.echo.qt import autoconnect_callbacks_to_qt
from glue.core import Data, Subset
from glue.utils.qt import load_ui
from glue.core.qt.data_combo_helper import ComponentIDComboHelper

__all__ = ['ScatterOptionsWidget']


class ScatterOptionsWidget(QtWidgets.QWidget):

    def __init__(self, viewer_state, session, parent=None):

        super(ScatterOptionsWidget, self).__init__(parent=parent)

        self.ui = load_ui('options_widget.ui', self,
                          directory=os.path.dirname(__file__))

        autoconnect_callbacks_to_qt(viewer_state, self.ui)

        viewer_state.add_callback('layers', self._update_combo_data)

        self.xatt_helper = ComponentIDComboHelper(self.ui.combodata_x_att,
                                                  session.data_collection,
                                                  default_index=0)

        self.yatt_helper = ComponentIDComboHelper(self.ui.combodata_y_att,
                                                  session.data_collection,
                                                  default_index=1)

        self.viewer_state = viewer_state

    def reset_limits(self):
        self.viewer_state.reset_limits()

    def _update_combo_data(self, *args):

        layers = []

        for layer_state in self.viewer_state.layers:
            if isinstance(layer_state.layer, Data):
                if layer_state.layer not in layers:
                    layers.append(layer_state.layer)

        for layer_state in self.viewer_state.layers:
            if isinstance(layer_state.layer, Subset) and layer_state.layer.data not in layers:
                if layer_state.layer not in layers:
                    layers.append(layer_state.layer)

        self.xatt_helper.set_multiple_data(layers)
        self.yatt_helper.set_multiple_data(layers)
