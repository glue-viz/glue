from __future__ import absolute_import, division, print_function

import os

from qtpy import QtWidgets

from glue.external.echo.qt import autoconnect_callbacks_to_qt
from glue.utils.qt import load_ui, fix_tab_widget_fontsize
from glue.viewers.matplotlib.state import MatplotlibDataViewerState

__all__ = ['ScatterOptionsWidget']


class ScatterOptionsWidget(QtWidgets.QWidget):

    def __init__(self, viewer_state, session, parent=None):

        super(ScatterOptionsWidget, self).__init__(parent=parent)

        self.ui = load_ui('options_widget.ui', self,
                          directory=os.path.dirname(__file__))

        fix_tab_widget_fontsize(self.ui.tab_widget)

        autoconnect_callbacks_to_qt(viewer_state, self.ui)
        autoconnect_callbacks_to_qt(viewer_state, self.ui.axes_editor.ui)

        self.viewer_state = viewer_state

        viewer_state.add_callback('x_att', self._update_x_attribute)
        viewer_state.add_callback('y_att', self._update_y_attribute)

        self.session = session
        self.ui.axes_editor.button_apply_all.clicked.connect(self._apply_all_viewers)

    def _apply_all_viewers(self):
        for tab in self.session.application.viewers:
            for viewer in tab:
                if isinstance(viewer.state, MatplotlibDataViewerState):
                    viewer.state.update_axes_settings_from(self.viewer_state)

    def _update_x_attribute(self, *args):
        # If at least one of the components is categorical or a date, disable log button
        log_enabled = 'categorical' not in self.viewer_state.x_kinds
        self.ui.bool_x_log.setEnabled(log_enabled)
        self.ui.bool_x_log_.setEnabled(log_enabled)
        if not log_enabled:
            self.ui.bool_x_log.setChecked(False)
            self.ui.bool_x_log_.setChecked(False)

    def _update_y_attribute(self, *args):
        # If at least one of the components is categorical or a date, disable log button
        log_enabled = 'categorical' not in self.viewer_state.y_kinds
        self.ui.bool_y_log.setEnabled(log_enabled)
        self.ui.bool_y_log_.setEnabled(log_enabled)
        if not log_enabled:
            self.ui.bool_y_log.setChecked(False)
            self.ui.bool_y_log_.setChecked(False)
