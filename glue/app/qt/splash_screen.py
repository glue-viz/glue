from __future__ import absolute_import, division, print_function

import os

from qtpy import QtWidgets, QtGui
from qtpy.QtCore import Qt, QRect

__all__ = ['QtSplashScreen']


class QtSplashScreen(QtWidgets.QWidget):

    def __init__(self, *args, **kwargs):

        super(QtSplashScreen, self).__init__(*args, **kwargs)

        self.resize(627, 310)
        self.setStyleSheet("background-color:white;")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        self.center()

        self.progress = QtWidgets.QProgressBar()

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addStretch()
        self.layout.addWidget(self.progress)

        pth = os.path.join(os.path.dirname(__file__), '..', '..', 'logo.png')
        self.image = QtGui.QPixmap(pth)

    def set_progress(self, value):
        self.progress.setValue(value)
        QtWidgets.qApp.processEvents()  # update progress bar

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawPixmap(QRect(20, 20, 587, 229), self.image)

    def center(self):
        # Adapted from StackOverflow
        # https://stackoverflow.com/questions/20243637/pyqt4-center-window-on-active-screen
        frameGm = self.frameGeometry()
        screen = QtWidgets.QApplication.desktop().screenNumber(QtWidgets.QApplication.desktop().cursor().pos())
        centerPoint = QtWidgets.QApplication.desktop().screenGeometry(screen).center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())
