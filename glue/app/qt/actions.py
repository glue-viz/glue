from qtpy import QtWidgets
from glue.icons.qt import get_icon


class GlueActionButton(QtWidgets.QPushButton):

    def set_action(self, action, text=True):
        self._text = text
        self._action = action
        self.clicked.connect(action.trigger)
        action.changed.connect(self._sync_to_action)
        self._sync_to_action()

    def _sync_to_action(self):
        self.setIcon(self._action.icon())
        if self._text:
            self.setText(self._action.text())
        self.setToolTip(self._action.toolTip())
        self.setWhatsThis(self._action.whatsThis())
        self.setEnabled(self._action.isEnabled())


def action(name, parent, tip='', icon=None, shortcut=None):
    """ Factory for making a new action """
    a = QtWidgets.QAction(name, parent)
    a.setToolTip(tip)
    if icon:
        a.setIcon(get_icon(icon))
    if shortcut:
        a.setShortcut(shortcut)
    return a
