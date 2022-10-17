import time
import platform
from qtpy import QtCore, QtGui, QtWidgets

from glue.config import settings
from glue._settings_helpers import save_settings

__all__ = ['process_events', 'get_qapp', 'fix_tab_widget_fontsize', 'update_global_font_size']

qapp = None


def __get_font_size_offset():
    if platform.system() == 'Darwin':
        # On Mac, the fonts are generally too large compared to other
        # applications, so we reduce the default here. In future we should
        # make this a setting in the system preferences.
        size_offset = 2
    else:
        # On other platforms, we reduce the font size by 1 point to save
        # space too. Again, this should probably be a global setting.
        size_offset = 1
    return size_offset


def process_events(wait=None):
    app = get_qapp()
    if wait is None:
        app.processEvents()
    else:
        start = time.time()
        while time.time() - start < wait:
            app.processEvents()


def get_qapp(icon_path=None):

    global qapp

    qapp = QtWidgets.QApplication.instance()

    if qapp is None:

        # NOTE: plugins that need WebEngine may complain that QtWebEngineWidgets
        # needs to be imported before QApplication is constructed, but this can
        # cause segmentation faults to crop up under certain conditions, so we
        # don't do it here and instead ask that the plugins do it in their
        # main __init__.py (which should get executed before glue is launched).

        qapp = QtWidgets.QApplication([''])
        qapp.setQuitOnLastWindowClosed(True)

        if icon_path is not None:
            qapp.setWindowIcon(QtGui.QIcon(icon_path))

        size_offset = __get_font_size_offset()
        font = qapp.font()

        if settings.FONT_SIZE is None or settings.FONT_SIZE == -1:
            default_font = QtGui.QFont()
            settings.FONT_SIZE = default_font.pointSize()
            save_settings()

        point_size = settings.FONT_SIZE
        font.setPointSize(int(point_size - size_offset))
        qapp.setFont(font)

    # Make sure we use high resolution icons for HDPI displays.
    try:
        qapp.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    except AttributeError:  # PyQt6/PySide6 don't have this setting as it is default
        pass

    return qapp


def fix_tab_widget_fontsize(tab_widget):
    """
    Because of a bug in Qt, tab titles on MacOS X don't have the right font size
    """
    if platform.system() == 'Darwin':
        app = get_qapp()
        app_font = app.font()
        tab_widget.setStyleSheet('font-size: {0}px'.format(app_font.pointSize()))


def update_global_font_size():
    """Updates the global font size through the current UI backend
    """
    if qapp is None:
        get_qapp()

    font = qapp.font()
    point_size = settings.FONT_SIZE
    size_offset = __get_font_size_offset()
    font.setPointSize(int(point_size - size_offset))
    qapp.setFont(font)
