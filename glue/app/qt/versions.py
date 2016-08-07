import os

from glue import __version__

from qtpy import QtWidgets
from qtpy.QtCore import Qt

from glue.utils import nonpartial
from glue.utils.qt import load_ui
from glue._deps import get_status_as_odict

__all__ = ['show_glue_info', 'QVersionsDialog']


class QVersionsDialog(QtWidgets.QDialog):

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
            check = QtWidgets.QTreeWidgetItem(self.ui.version_tree.invisibleRootItem(),
                                          [name, version])
            self._text += "{0}: {1}\n".format(name, version)

    def _copy(self):
        self._clipboard.setText(self._text)

    def center(self):
        # Adapted from StackOverflow
        # http://stackoverflow.com/questions/20243637/pyqt4-center-window-on-active-screen
        frameGm = self.frameGeometry()
        screen = QtWidgets.QApplication.desktop().screenNumber(QtWidgets.QApplication.desktop().cursor().pos())
        centerPoint = QtWidgets.QApplication.desktop().screenGeometry(screen).center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())


def show_glue_info():
    window = QVersionsDialog()
    window.show()
    window.exec_()

if __name__ == "__main__":

    from glue.utils.qt import get_qapp
    app = get_qapp()
    show_glue_info()
