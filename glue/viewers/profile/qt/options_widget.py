import os

from qtpy import QtWidgets

from echo.qt import autoconnect_callbacks_to_qt
from glue.utils.qt import load_ui, fix_tab_widget_fontsize
from glue.viewers.profile.qt.slice_widget import ProfileMultiSliceWidgetHelper
from glue.viewers.matplotlib.state import MatplotlibDataViewerState

__all__ = ['ProfileOptionsWidget']


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

        self.profile_slice_helper = ProfileMultiSliceWidgetHelper(viewer_state=self.viewer_state,
                                                                  session=self.session,
                                                                  layout=self.ui.layout_slices)

        self.ui.text_warning.hide()

        self.ui.axes_editor.button_apply_all.clicked.connect(self._apply_all_viewers)

    def _on_attribute_change(self, *args):

        if (self.viewer_state.reference_data is None or
                self.viewer_state.x_att_pixel is None or
                self.viewer_state.x_att is self.viewer_state.x_att_pixel):
            self.ui.text_warning.hide()
            return

        self.ui.text_warning.hide()
        self.ui.text_warning.setText('')

    def _apply_all_viewers(self):
        for tab in self.session.application.viewers:
            for viewer in tab:
                if isinstance(viewer.state, MatplotlibDataViewerState):
                    viewer.state.update_axes_settings_from(self.viewer_state)
