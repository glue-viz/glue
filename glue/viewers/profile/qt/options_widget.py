from __future__ import absolute_import, division, print_function

import os

from qtpy import QtWidgets

from glue.external.echo.qt import autoconnect_callbacks_to_qt
from glue.utils.qt import load_ui, fix_tab_widget_fontsize

__all__ = ['ProfileOptionsWidget']


WARNING_TEXT = ("Warning: the coordinate '{label}' is not aligned with pixel "
                "grid for any of the datasets, so no profiles could be "
                "computed. Try selecting another world coordinates or one of the "
                "pixel coordinates.")


class ProfileOptionsWidget(QtWidgets.QWidget):

    def __init__(self, viewer_state, session, parent=None):

        super(ProfileOptionsWidget, self).__init__(parent=parent)

        self.ui = load_ui('options_widget.ui', self,
                          directory=os.path.dirname(__file__))

        fix_tab_widget_fontsize(self.ui.tab_widget)

        autoconnect_callbacks_to_qt(viewer_state, self.ui)
        autoconnect_callbacks_to_qt(viewer_state, self.ui.axes_editor.ui)

        self.viewer_state = viewer_state

        self.session = session

        self.viewer_state.add_callback('x_att', self._on_attribute_change)

        self.ui.text_warning.hide()

    def _on_attribute_change(self, *args):

        if self.viewer_state.x_att is None:
            self.ui.text_warning.hide()
            return

        for layer_state in self.viewer_state.layers:
            if layer_state.independent_x_att:
                self.ui.text_warning.hide()
                return

        self.ui.text_warning.show()
        self.ui.text_warning.setText(WARNING_TEXT.format(label=self.viewer_state.x_att.label))
