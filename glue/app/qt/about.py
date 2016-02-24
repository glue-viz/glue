import os

from glue import __version__

from glue.external.qt import QtGui
from glue.external.qt.QtCore import Qt, QRect

__all__ = ['QAboutDialog']


class QAboutDialog(QtGui.QWidget):

    def __init__(self, *args, **kwargs):

        super(QAboutDialog, self).__init__(*args, **kwargs)

        self.resize(627, 269)
        self.setStyleSheet("background-color:white;");
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        self.center()

        self.text = QtGui.QLabel()

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addStretch()
        self.layout.addWidget(self.text)

        self.text.setText("Version {0}".format(__version__))
        self.text.setAlignment(Qt.AlignRight)
        self.text.setStyleSheet("background-color:none;");

    def paintEvent(self, event):

        pth = os.path.join(os.path.dirname(__file__), '..', '..', 'logo.png')
        image = QtGui.QPixmap(pth)

        painter = QtGui.QPainter(self)
        painter.drawPixmap(QRect(20, 20, 587, 229), image)

    def center(self):
        # Adapted from StackOverflow
        # http://stackoverflow.com/questions/20243637/pyqt4-center-window-on-active-screen
        frameGm = self.frameGeometry()
        screen = QtGui.QApplication.desktop().screenNumber(QtGui.QApplication.desktop().cursor().pos())
        centerPoint = QtGui.QApplication.desktop().screenGeometry(screen).center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())


if __name__ == "__main__":

    from glue.external.qt import get_qapp
    app = get_qapp()
    window = QAboutDialog()
    window.show()
    app.exec_()
