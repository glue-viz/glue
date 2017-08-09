import platform
from qtpy import QtCore, QtGui, QtWidgets, PYQT5

__all__ = ['get_qapp']

qapp = None


def get_qapp(icon_path=None):

    global qapp

    qapp = QtWidgets.QApplication.instance()

    if qapp is None:

        qapp = QtWidgets.QApplication([''])
        qapp.setQuitOnLastWindowClosed(True)

        if icon_path is not None:
            qapp.setWindowIcon(QtGui.QIcon(icon_path))

        if platform.system() == 'Darwin':
            # On Mac, the fonts are generally too large compared to other
            # applications, so we reduce the default here. In future we should
            # make this a setting in the system preferences.
            size_offset = 2
        else:
            # On other platforms, we reduce the font size by 1 point to save
            # space too. Again, this should probably be a global setting.
            size_offset = 1

        font = qapp.font()
        font.setPointSize(font.pointSize() - size_offset)
        qapp.setFont(font)

    # Make sure we use high resolution icons with PyQt5 for HDPI
    # displays.
    if PYQT5:
        qapp.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)

    return qapp
