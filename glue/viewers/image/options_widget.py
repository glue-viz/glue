from __future__ import absolute_import, division, print_function

import os

from qtpy import QtWidgets

from glue.core import Data
from glue.utils.qt import load_ui
from glue.core.qt.data_combo_helper import ComponentIDComboHelper
from glue.viewers.common.qt.attribute_limits_helper import AttributeLimitsHelper

from glue_new_viewers.common.qt_helpers import autoconnect_qt

__all__ = ['ImageOptionsWidget']


class ImageOptionsWidget(QtWidgets.QWidget):

    def __init__(self, viewer_state, session, parent=None):

        super(ImageOptionsWidget, self).__init__(parent=parent)

        self.ui = load_ui('options_widget.ui', self,
                          directory=os.path.dirname(__file__))

        autoconnect_qt(viewer_state, self.ui)

        self.att_helper = ComponentIDComboHelper(self.ui.combo_xcoord,
                                                  session.data_collection)

        # TODO: make it so there is an equivalent of ComponentIDComboHelper for
        # coordinates

        # self.xatt_helper = ComponentIDComboHelper(self.ui.combo_xatt,
        #                                           session.data_collection)
        # self.yatt_helper = ComponentIDComboHelper(self.ui.combo_yatt,
        #                                           session.data_collection)

        self.xcoord_limits_helper = AttributeLimitsHelper(self.ui.combo_xcoord,
                                                          self.ui.value_x_min,
                                                          self.ui.value_x_max,
                                                          flip_button=self.ui.button_flip_x)

        self.ycoord_limits_helper = AttributeLimitsHelper(self.ui.combo_ycoord,
                                                          self.ui.value_y_min,
                                                          self.ui.value_y_max,
                                                          flip_button=self.ui.button_flip_y)

        self.viewer_state = viewer_state
