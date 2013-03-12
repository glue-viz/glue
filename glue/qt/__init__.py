from ..external.qt.QtGui import QApplication, QIcon
from . import glue_qt_resources


def get_qapp():
    qapp = QApplication.instance()
    if qapp is None:
        qapp = QApplication([''])
        qapp.setQuitOnLastWindowClosed(True)
        qapp.setWindowIcon(QIcon(':icons/app_icon.png'))
    return qapp


def teardown():
    app = get_qapp()
    app.exit()

_app = get_qapp()
import atexit
atexit.register(teardown)
