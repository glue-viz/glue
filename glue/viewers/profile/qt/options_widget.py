import os

from qtpy import QtWidgets

from glue.core.coordinate_helpers import dependent_axes
from echo.qt import autoconnect_callbacks_to_qt
from glue.utils.qt import load_ui, fix_tab_widget_fontsize

__all__ = ['ProfileOptionsWidget']


WARNING_TEXT = ("Warning: the coordinate '{label}' is not aligned with pixel "
                "grid, so the values shown on the x-axis are approximate.")


class ProfileOptionsWidget(QtWidgets.QWidget):

    def __init__(self, viewer_state, session, parent=None):

        super(ProfileOptionsWidget, self).__init__(parent=parent)

        self.ui = load_ui('options_widget.ui', self,
                          directory=os.path.dirname(__file__))

        fix_tab_widget_fontsize(self.ui.tab_widget)

        self._connections = autoconnect_callbacks_to_qt(viewer_state, self.ui)
        self._connections_axes = autoconnect_callbacks_to_qt(viewer_state, self.ui.axes_editor.ui)
        connect_kwargs = {'alpha': dict(value_range=(0, 1))}
        self._connections_legend = autoconnect_callbacks_to_qt(viewer_state.legend, self.ui.legend_editor.ui, connect_kwargs)

        self.viewer_state = viewer_state

        self.session = session

        self.viewer_state.add_callback('x_att', self._on_attribute_change)

        self.ui.text_warning.hide()

    def _on_attribute_change(self, *args):

        if (self.viewer_state.reference_data is None or
                self.viewer_state.x_att_pixel is None or
                self.viewer_state.x_att is self.viewer_state.x_att_pixel):
            self.ui.text_warning.hide()
            return

        world_warning = len(dependent_axes(self.viewer_state.reference_data.coords,
                                           self.viewer_state.x_att_pixel.axis)) > 1

        if world_warning:
            self.ui.text_warning.show()
            self.ui.text_warning.setText(WARNING_TEXT.format(label=self.viewer_state.x_att.label))
        else:
            self.ui.text_warning.hide()
            self.ui.text_warning.setText('')
