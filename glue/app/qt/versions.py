import os

from glue import __version__

from qtpy import QtWidgets
from qtpy.QtCore import Qt

from glue.utils import nonpartial
from glue.utils.qt import load_ui, CenteredDialog
from glue._deps import get_status_as_odict

__all__ = ['QVersionsDialog']


class QVersionsDialog(CenteredDialog):

    def __init__(self, *args, **kwargs):

        super(QVersionsDialog, self).__init__(*args, **kwargs)

        self.ui = load_ui('versions.ui', self, directory=os.path.dirname(__file__))

        self.resize(400, 500)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        self.center()

        self._update_deps()

        self._clipboard = QtWidgets.QApplication.clipboard()
        self.ui.button_copy.clicked.connect(nonpartial(self._copy))

    def _update_deps(self):
        status = get_status_as_odict()
        self._text = ""
        for name, version in [('Glue', __version__)] + list(status.items()):
            QtWidgets.QTreeWidgetItem(self.ui.version_tree.invisibleRootItem(),
                                      [name, version])
            self._text += "{0}: {1}\n".format(name, version)

    def _copy(self):
        self._clipboard.setText(self._text)


if __name__ == "__main__":

    from glue.utils.qt import get_qapp
    app = get_qapp()
    window = QVersionsDialog()
    window.show()
    window.exec_()
