from __future__ import absolute_import, division, print_function

import os

from qtpy import QtWidgets

from glue.external.echo.qt import autoconnect_callbacks_to_qt
from glue.utils.qt import load_ui
from glue.core.qt.data_combo_helper import QtComboHelper
from glue.viewers.image.qt.slice_widget import MultiSliceWidgetHelper

__all__ = ['ImageOptionsWidget']


class ImageOptionsWidget(QtWidgets.QWidget):

    def __init__(self, viewer_state, session, parent=None):

        super(ImageOptionsWidget, self).__init__(parent=parent)

        self.ui = load_ui('options_widget.ui', self,
                          directory=os.path.dirname(__file__))

        self.ui.combodata_aspect.addItem("Square Pixels", userData='equal')
        self.ui.combodata_aspect.addItem("Automatic", userData='auto')
        self.ui.combodata_aspect.setCurrentIndex(0)

        self.ui.combotext_color_mode.addItem("Colormaps")
        self.ui.combotext_color_mode.addItem("One color per layer")

        autoconnect_callbacks_to_qt(viewer_state, self.ui)

        self.ref_data_helper = QtComboHelper(self.ui.combonew_reference_data,
                                             viewer_state, 'reference_data',
                                             '_reference_data_choices')
        self.x_att_world_helper = QtComboHelper(self.ui.combonew_x_att_world,
                                                viewer_state, 'x_att_world',
                                                '_x_att_world_choices')
        self.y_att_world_helper = QtComboHelper(self.ui.combonew_y_att_world,
                                                viewer_state, 'y_att_world',
                                                '_y_att_world_choices')

        self.viewer_state = viewer_state

        self.slice_helper = MultiSliceWidgetHelper(viewer_state=self.viewer_state,
                                                   widget=self.ui.slice_tab)
