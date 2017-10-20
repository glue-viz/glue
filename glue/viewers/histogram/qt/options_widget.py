from __future__ import absolute_import, division, print_function

import os

from qtpy import QtWidgets

from glue.external.echo.qt import autoconnect_callbacks_to_qt
from glue.utils import nonpartial
from glue.utils.qt import load_ui, fix_tab_widget_fontsize

__all__ = ['HistogramOptionsWidget']


class HistogramOptionsWidget(QtWidgets.QWidget):

    def __init__(self, viewer_state, session, parent=None):

        super(HistogramOptionsWidget, self).__init__(parent=parent)

        self.ui = load_ui('options_widget.ui', self,
                          directory=os.path.dirname(__file__))

        fix_tab_widget_fontsize(self.ui.tab_widget)

        autoconnect_callbacks_to_qt(viewer_state, self.ui)

        self.viewer_state = viewer_state

        viewer_state.add_callback('x_att', nonpartial(self._update_attribute))

    def _update_attribute(self):
        # If at least one of the components is categorical, disable log button
        log_enabled = not any(comp.categorical for comp in self.viewer_state._get_x_components())
        self.ui.bool_x_log.setEnabled(log_enabled)
        if not log_enabled:
            self.ui.bool_x_log.setChecked(False)
