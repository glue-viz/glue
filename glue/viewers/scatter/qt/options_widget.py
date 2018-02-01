from __future__ import absolute_import, division, print_function

import os

from qtpy import QtWidgets

from glue.external.echo.qt import autoconnect_callbacks_to_qt
from glue.utils.qt import load_ui, fix_tab_widget_fontsize

__all__ = ['ScatterOptionsWidget']


class ScatterOptionsWidget(QtWidgets.QWidget):

    def __init__(self, viewer_state, session, parent=None):

        super(ScatterOptionsWidget, self).__init__(parent=parent)

        self.ui = load_ui('options_widget.ui', self,
                          directory=os.path.dirname(__file__))

        fix_tab_widget_fontsize(self.ui.tab_widget)

        autoconnect_callbacks_to_qt(viewer_state, self.ui)

        self.viewer_state = viewer_state

        viewer_state.add_callback('x_att', self._update_x_attribute)
        viewer_state.add_callback('y_att', self._update_y_attribute)

    def _update_x_attribute(self, *args):
        # If at least one of the components is categorical or a date, disable log button
        log_enabled = not any(comp.categorical or comp.date for comp in self.viewer_state._get_x_components())
        self.ui.bool_x_log.setEnabled(log_enabled)
        self.ui.bool_x_log_.setEnabled(log_enabled)
        if not log_enabled:
            self.ui.bool_x_log.setChecked(False)
            self.ui.bool_x_log_.setChecked(False)

    def _update_y_attribute(self, *args):
        # If at least one of the components is categorical or a date, disable log button
        log_enabled = not any(comp.categorical or comp.date for comp in self.viewer_state._get_y_components())
        self.ui.bool_y_log.setEnabled(log_enabled)
        self.ui.bool_y_log_.setEnabled(log_enabled)
        if not log_enabled:
            self.ui.bool_y_log.setChecked(False)
            self.ui.bool_y_log_.setChecked(False)
