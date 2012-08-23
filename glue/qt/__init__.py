try:
    from PyQt4.QtGui import QApplication
except ImportError:
    raise ImportError("PyQt4 is required for using GUI features of Glue")

qapp = QApplication.instance()
if qapp is None:
    import sys
    qapp = QApplication(sys.argv)
