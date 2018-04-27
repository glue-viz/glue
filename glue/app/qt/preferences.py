from __future__ import absolute_import, division, print_function

import os
import weakref
import platform

import numpy as np
from matplotlib.colors import ColorConverter

from qtpy import QtWidgets
from glue.core.message import SettingsChangeMessage
from glue.utils.qt import load_ui, ColorProperty, get_qapp
from glue.utils.qt.widget_properties import (CurrentComboTextProperty,
                                             ValueProperty, ButtonProperty)
from glue._settings_helpers import save_settings

__all__ = ["PreferencesDialog"]

rgb = ColorConverter().to_rgb


class PreferencesDialog(QtWidgets.QDialog):

    theme = CurrentComboTextProperty('ui.combo_theme')
    background = ColorProperty('ui.color_background')
    foreground = ColorProperty('ui.color_foreground')
    data_color = ColorProperty('ui.color_default_data')
    data_alpha = ValueProperty('ui.slider_alpha', value_range=(0, 1))
    data_apply = ButtonProperty('ui.checkbox_apply')
    save_to_disk = ButtonProperty('ui.checkbox_save')

    def __init__(self, application, parent=None):

        super(PreferencesDialog, self).__init__(parent=parent)

        self._app = weakref.ref(application)

        self.ui = load_ui('preferences.ui', self,
                          directory=os.path.dirname(__file__))

        self.ui.cancel.clicked.connect(self.reject)
        self.ui.ok.clicked.connect(self.accept)

        self.ui.combo_theme.currentIndexChanged.connect(self._update_colors_from_theme)

        self.ui.button_reset_dialogs.clicked.connect(self._reset_dialogs)

        # The following is needed because of a bug in Qt which means that
        # tab titles don't get scaled right.
        if platform.system() == 'Darwin':
            app = get_qapp()
            app_font = app.font()
            self.ui.tab_widget.setStyleSheet('font-size: {0}px'.format(app_font.pointSize()))

        from glue.config import settings
        self.background = settings.BACKGROUND_COLOR
        self.foreground = settings.FOREGROUND_COLOR
        self.data_color = settings.DATA_COLOR
        self.data_alpha = settings.DATA_ALPHA

        self._update_theme_from_colors()

        self.panes = []

        from glue.config import preference_panes
        for label, widget_cls in sorted(preference_panes):
            pane = widget_cls()
            self.ui.tab_widget.addTab(pane, label)
            self.panes.append(pane)

    def _update_theme_from_colors(self, *args):
        if (rgb(self.background) == (1, 1, 1) and rgb(self.foreground) == (0, 0, 0) and
                rgb(self.data_color) == (0.35, 0.35, 0.35) and np.allclose(self.data_alpha, 0.8)):
            self.theme = 'Black on White'
        elif (rgb(self.background) == (0, 0, 0) and rgb(self.foreground) == (1, 1, 1) and
                rgb(self.data_color) == (0.75, 0.75, 0.75) and np.allclose(self.data_alpha, 0.8)):
            self.theme = 'White on Black'
        else:
            self.theme = 'Custom'

    def _update_colors_from_theme(self, *args):
        if self.theme == 'Black on White':
            self.foreground = 'black'
            self.background = 'white'
            self.data_color = '0.35'
            self.data_alpha = 0.8
        elif self.theme == 'White on Black':
            self.foreground = 'white'
            self.background = 'black'
            self.data_color = '0.75'
            self.data_alpha = 0.8
        elif self.theme != 'Custom':
            raise ValueError("Unknown theme: {0}".format(self.theme))

    def _reset_dialogs(self, *args):
        from glue.config import settings
        for key, _, _ in settings:
            if key.lower().startswith(('show_info', 'show_warn', 'show_large')):
                setattr(settings, key, True)

    def accept(self):

        # Update default settings

        from glue.config import settings
        settings.FOREGROUND_COLOR = self.foreground
        settings.BACKGROUND_COLOR = self.background
        settings.DATA_COLOR = self.data_color
        settings.DATA_ALPHA = self.data_alpha

        for pane in self.panes:
            pane.finalize()

        # Save to disk if requested
        if self.save_to_disk:
            save_settings()
        else:
            settings._save_to_disk = True

        # Trigger viewers to update defaults

        app = self._app()

        if app is not None:
            app._hub.broadcast(SettingsChangeMessage(self, ('FOREGROUND_COLOR', 'BACKGROUND_COLOR')))
            if self.data_apply:  # If requested, trigger data to update color
                app.set_data_color(settings.DATA_COLOR, settings.DATA_ALPHA)

        super(PreferencesDialog, self).accept()


if __name__ == "__main__":

    app = get_qapp()
    widget = PreferencesDialog()
    widget.show()
    widget.raise_()
    app.exec_()
