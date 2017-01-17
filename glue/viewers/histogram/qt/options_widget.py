from __future__ import absolute_import, division, print_function

import os

from qtpy import QtWidgets

from glue.core import Data
from glue.external.echo.qt import autoconnect_callbacks_to_qt
from glue.utils.qt import load_ui
from glue.core.qt.data_combo_helper import ComponentIDComboHelper
from glue.viewers.common.qt.attribute_limits_helper import AttributeLimitsHelper

__all__ = ['HistogramOptionsWidget']


class HistogramOptionsWidget(QtWidgets.QWidget):

    def __init__(self, viewer_state, session, parent=None):

        super(HistogramOptionsWidget, self).__init__(parent=parent)

        self.ui = load_ui('options_widget.ui', self,
                          directory=os.path.dirname(__file__))

        autoconnect_callbacks_to_qt(viewer_state, self.ui)

        viewer_state.connect('layers', self._update_combo_data)

        self.xatt_helper = ComponentIDComboHelper(self.ui.combodata_xatt,
                                                  session.data_collection)

        self.xatt_limits_helper = AttributeLimitsHelper(self.ui.combodata_xatt,
                                                        self.ui.value_x_min,
                                                        self.ui.value_x_max,
                                                        flip_button=self.ui.button_flip_x,
                                                        log_button=self.ui.bool_log_x)

        self.viewer_state = viewer_state

    def _update_combo_data(self, *args):
        # TODO: what about if only subset and not data is present?
        layers = [layer_state.layer for layer_state in self.viewer_state.layers
                  if isinstance(layer_state.layer, Data)]
        self.xatt_helper.set_multiple_data(layers)
