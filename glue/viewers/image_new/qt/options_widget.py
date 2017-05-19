from __future__ import absolute_import, division, print_function

import os

from qtpy import QtWidgets

from glue.external.echo.qt import autoconnect_callbacks_to_qt
from glue.utils.qt import load_ui
from glue.core.qt.data_combo_helper import ComponentIDComboHelper
from glue.viewers.image_new.qt.slice_widget import MultiSliceWidgetHelper

__all__ = ['ImageOptionsWidget']


class ImageOptionsWidget(QtWidgets.QWidget):

    def __init__(self, viewer_state, session, parent=None):

        super(ImageOptionsWidget, self).__init__(parent=parent)

        self.ui = load_ui('options_widget.ui', self,
                          directory=os.path.dirname(__file__))

        autoconnect_callbacks_to_qt(viewer_state, self.ui)

        viewer_state.add_callback('reference_data', self._update_combo_data)

        self.ui.combodata_aspect.addItem("Square Pixels", userData='equal')
        self.ui.combodata_aspect.addItem("Automatic", userData='auto')
        self.ui.combodata_aspect.setCurrentIndex(0)

        self.x_att_helper = ComponentIDComboHelper(self.ui.combodata_x_att,
                                                   session.data_collection,
                                                   numeric=False, categorical=False,
                                                   visible=False, pixel_coord=True, default_index=0)

        self.y_att_helper = ComponentIDComboHelper(self.ui.combodata_y_att,
                                                   session.data_collection,
                                                   numeric=False, categorical=False,
                                                   visible=False, pixel_coord=True, default_index=1)

        self.viewer_state = viewer_state

        self.slice_helper = MultiSliceWidgetHelper(viewer_state=self.viewer_state, widget=self.ui.slice_tab)

    def _update_combo_data(self, *args):
        self.x_att_helper.set_multiple_data([self.viewer_state.reference_data])
        self.y_att_helper.set_multiple_data([self.viewer_state.reference_data])
