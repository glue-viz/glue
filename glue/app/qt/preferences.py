from __future__ import absolute_import, division, print_function

import os

from matplotlib.colors import ColorConverter

from glue.external.qt.QtCore import Qt
from glue.external.qt import QtGui
from glue.config import settings
from glue._plugin_helpers import PluginConfig
from glue.utils import nonpartial
from glue.utils.qt import load_ui, ColorProperty
from glue.utils.qt.widget_properties import CurrentComboTextProperty

__all__ = ["Preferences"]

color_to_rgb = ColorConverter.to_rgb


class Preferences(QtGui.QDialog):

    theme = CurrentComboTextProperty('ui.combo_theme')
    background = ColorProperty('ui.color_background')
    foreground = ColorProperty('ui.color_foreground')
    data_color = ColorProperty('ui.color_default_data')

    def __init__(self, parent=None):

        super(Preferences, self).__init__(parent=parent)

        self.ui = load_ui('preferences.ui', self,
                           directory=os.path.dirname(__file__))

        self.ui.cancel.clicked.connect(self.reject)
        self.ui.ok.clicked.connect(self.finalize)

        self.ui.combo_theme.currentIndexChanged.connect(nonpartial(self._update_colors_from_theme))

        self.ui.color_foreground.mousePressed.connect(nonpartial(self._update_theme_from_colors))
        self.ui.color_background.mousePressed.connect(nonpartial(self._update_theme_from_colors))

        self.background = settings.BACKGROUND_COLOR
        self.foreground = settings.FOREGROUND_COLOR
        self.data_color = settings.DATA_COLOR

        self._update_theme_from_colors()

    def _update_theme_from_colors(self):
        if self.ui.color_background.to_rgb() == (1,1,1) and self.ui.color_foreground.to_rgb() == (0,0,0):
            self.theme = 'Black on White'
        elif self.ui.color_background.to_rgb() == (0,0,0) and self.ui.color_foreground.to_rgb() == (1,1,1):
            self.theme = 'White on Black'
        else:
            self.theme = 'Custom'

    def _update_colors_from_theme(self):
        if self.theme == 'Black on White':
            self.foreground = 'black'
            self.background = 'white'
        elif self.theme == 'White on Black':
            self.foreground = 'white'
            self.background = 'black'
        elif self.theme != 'Custom':
            raise ValueError("Unknown theme: {0}".format(self.theme))

    def finalize(self):

        self.ui.accept()


if __name__ == "__main__":

    from glue.external.qt import get_qapp
    app = get_qapp()
    widget = Preferences()
    widget.show()
    widget.raise_()
    app.exec_()
