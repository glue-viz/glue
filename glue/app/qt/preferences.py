from __future__ import absolute_import, division, print_function

import os

from matplotlib.colors import ColorConverter

from glue.external.qt import QtGui
from glue.config import settings
from glue.utils import nonpartial
from glue.utils.qt import load_ui, ColorProperty
from glue.utils.qt.widget_properties import CurrentComboTextProperty, ValueProperty

__all__ = ["PreferencesDialog"]

rgb = ColorConverter().to_rgb


class PreferencesDialog(QtGui.QDialog):

    theme = CurrentComboTextProperty('ui.combo_theme')
    background = ColorProperty('ui.color_background')
    foreground = ColorProperty('ui.color_foreground')
    data_color = ColorProperty('ui.color_default_data')
    data_alpha = ValueProperty('ui.slider_alpha', value_range=(0, 1))

    def __init__(self, parent=None):

        super(PreferencesDialog, self).__init__(parent=parent)

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
        self.data_alpha = float(settings.DATA_ALPHA)

        self._update_theme_from_colors()

    def _update_theme_from_colors(self):
        if rgb(self.background) == (1,1,1) and rgb(self.foreground) == (0,0,0):
            self.theme = 'Black on White'
        elif rgb(self.background) == (0,0,0) and rgb(self.foreground) == (1,1,1):
            self.theme = 'White on Black'
        else:
            self.theme = 'Custom'

    def _update_colors_from_theme(self):
        if self.theme == 'Black on White':
            self.foreground = 'black'
            self.background = 'white'
            self.data_color = '0.25'
            self.data_alpha = 0.75
        elif self.theme == 'White on Black':
            self.foreground = 'white'
            self.background = 'black'
            self.data_color = '0.75'
            self.data_alpha = 0.75
        elif self.theme != 'Custom':
            raise ValueError("Unknown theme: {0}".format(self.theme))

    def finalize(self):

        settings.FOREGROUND_COLOR = self.foreground
        settings.BACKGROUND_COLOR = self.background
        settings.DATA_COLOR = self.data_color
        settings.DATA_ALPHA = self.data_alpha

        self.ui.accept()


if __name__ == "__main__":

    from glue.external.qt import get_qapp
    app = get_qapp()
    widget = PreferencesDialog()
    widget.show()
    widget.raise_()
    app.exec_()
