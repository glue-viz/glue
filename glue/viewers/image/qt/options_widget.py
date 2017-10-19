from __future__ import absolute_import, division, print_function

import os

from qtpy import QtWidgets

from glue.external.echo.qt import autoconnect_callbacks_to_qt
from glue.utils.qt import load_ui, fix_tab_widget_fontsize
from glue.viewers.image.qt.slice_widget import MultiSliceWidgetHelper

__all__ = ['ImageOptionsWidget']


class ImageOptionsWidget(QtWidgets.QWidget):

    def __init__(self, viewer_state, session, parent=None):

        super(ImageOptionsWidget, self).__init__(parent=parent)

        self.ui = load_ui('options_widget.ui', self,
                          directory=os.path.dirname(__file__))

        fix_tab_widget_fontsize(self.ui.tab_widget)

        self.ui.combodata_aspect.addItem("Square Pixels", userData='equal')
        self.ui.combodata_aspect.addItem("Automatic", userData='auto')
        self.ui.combodata_aspect.setCurrentIndex(0)

        self.ui.combotext_color_mode.addItem("Colormaps")
        self.ui.combotext_color_mode.addItem("One color per layer")

        autoconnect_callbacks_to_qt(viewer_state, self.ui)

        self.viewer_state = viewer_state

        self.slice_helper = MultiSliceWidgetHelper(viewer_state=self.viewer_state,
                                                   layout=self.ui.layout_slices)
