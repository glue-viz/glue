from __future__ import absolute_import, division, print_function

import os

from qtpy import QtWidgets

from glue.external.echo.qt import autoconnect_callbacks_to_qt
from glue.core import Data, Subset
from glue.utils.qt import load_ui
from glue.core.qt.data_combo_helper import ComponentIDComboHelper

__all__ = ['ScatterOptionsWidget']


class ScatterOptionsWidget(QtWidgets.QWidget):

    def __init__(self, viewer_state, session, parent=None):

        super(ScatterOptionsWidget, self).__init__(parent=parent)

        self.ui = load_ui('options_widget.ui', self,
                          directory=os.path.dirname(__file__))

        autoconnect_callbacks_to_qt(viewer_state, self.ui)

        self.viewer_state = viewer_state

    def reset_limits(self):
        self.viewer_state.reset_limits()
