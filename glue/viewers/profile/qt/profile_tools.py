import os

from qtpy import QtWidgets
from glue.utils.qt import load_ui, fix_tab_widget_fontsize
from glue.viewers.profile.mouse_mode import NavigateMouseMode, RangeMouseMode


__all__ = ['ProfileTools']


MODES = ['navigate', 'fit', 'collapse']


class ProfileTools(QtWidgets.QWidget):

    def __init__(self, parent=None):

        super(ProfileTools, self).__init__(parent=parent)

        self.ui = load_ui('profile_tools.ui', self,
                          directory=os.path.dirname(__file__))

        fix_tab_widget_fontsize(self.ui.tabs)

        self.viewer = parent

    def enable(self):
        self.nav_mode = NavigateMouseMode(self.viewer)
        self.rng_mode = RangeMouseMode(self.viewer)
        self.ui.tabs.setCurrentIndex(0)
        self.ui.tabs.currentChanged.connect(self._on_tab_change)
        self._on_tab_change()

    @property
    def mode(self):
        return MODES[self.tabs.currentIndex()]

    def _on_tab_change(self, *event):
        mode = self.mode
        if mode == 'navigate':
            self.rng_mode.deactivate()
            self.nav_mode.activate()
        else:
            self.rng_mode.activate()
            self.nav_mode.deactivate()
