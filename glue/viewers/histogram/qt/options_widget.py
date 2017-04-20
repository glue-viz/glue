from __future__ import absolute_import, division, print_function

import os

from qtpy import QtWidgets

from glue.core import Data
from glue.external.echo.qt import autoconnect_callbacks_to_qt
from glue.utils import nonpartial
from glue.utils.qt import load_ui
from glue.core.qt.data_combo_helper import ComponentIDComboHelper

__all__ = ['HistogramOptionsWidget']


class HistogramOptionsWidget(QtWidgets.QWidget):

    def __init__(self, viewer_state, session, parent=None):

        super(HistogramOptionsWidget, self).__init__(parent=parent)

        self.ui = load_ui('options_widget.ui', self,
                          directory=os.path.dirname(__file__))

        autoconnect_callbacks_to_qt(viewer_state, self.ui)

        viewer_state.add_callback('layers', self._update_combo_data)

        self.x_att_helper = ComponentIDComboHelper(self.ui.combodata_x_att,
                                                  session.data_collection)

        self.viewer_state = viewer_state

        viewer_state.add_callback('x_att', nonpartial(self._update_attribute))

    def _update_attribute(self):
        # If at least one of the components is categorical, disable log button
        log_enabled = not any(comp.categorical for comp in self.viewer_state._get_x_components())
        self.ui.bool_x_log.setEnabled(log_enabled)
        if not log_enabled:
            self.ui.bool_x_log.setChecked(False)

    def _update_combo_data(self, *args):
        # TODO: what about if only subset and not data is present?
        layers = [layer_state.layer for layer_state in self.viewer_state.layers
                  if isinstance(layer_state.layer, Data)]
        self.x_att_helper.set_multiple_data(layers)
