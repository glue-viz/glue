from ..external.qt.QtGui import QApplication, QIcon
import os


def get_qapp():
    qapp = QApplication.instance()
    if qapp is None:
        qapp = QApplication([''])
        qapp.setQuitOnLastWindowClosed(True)
        pth = os.path.abspath(os.path.dirname(__file__))
        pth = os.path.join(pth, 'icons', 'app_icon.png')
        qapp.setWindowIcon(QIcon(pth))
    return qapp


def teardown():
    app = get_qapp()
    app.exit()

_app = get_qapp()
import atexit
atexit.register(teardown)
