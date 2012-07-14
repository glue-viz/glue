#!/usr/bin/env python
import sys
import sip
sip.setapi('QVariant', 2)
sip.setapi('QString', 2)

from PyQt4.QtGui import QApplication

from glue.qt.glue_application import GlueApplication

def main():
    app = QApplication(sys.argv)
    win = GlueApplication()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()