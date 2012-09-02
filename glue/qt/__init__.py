try:
    from PyQt4.QtGui import QApplication
except ImportError:
    raise ImportError("PyQt4 is required for using GUI features of Glue")

def get_qapp():
    qapp = QApplication.instance()
    if qapp is None:
        import sys
        qapp = QApplication(sys.argv)
    return qapp

def teardown():
    app = get_qapp()
    app.exit()

_app = get_qapp()
import atexit
atexit.register(teardown)
