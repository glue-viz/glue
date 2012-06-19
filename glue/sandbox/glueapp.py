from PyQt4.QtGui import QApplication
import sys
import glue
from glue.qt.messagewidget import MessageWidget
from glue.qt.glue_application import GlueApplication

def print_widget(widget):
    print widget

def main():
    app = QApplication(sys.argv)
    #app.focusChanged.connect(lambda x,y: print_widget(y))
    win = GlueApplication()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()