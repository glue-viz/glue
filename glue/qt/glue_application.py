from PyQt4.QtCore import Qt, SIGNAL
from PyQt4.QtGui import *

from ui_glue_application import Ui_GlueApplication
from glue_viz_loader_widget import GlueVizLoaderWidget
import glue
from glue.message import ErrorMessage

class GlueApplication(QMainWindow, glue.HubListener):

    def __init__(self):
        QMainWindow.__init__(self)
        self._ui = Ui_GlueApplication()
        self._ui.setupUi(self)
        self._ui.layerWidget.set_checkable(False)
        self._data = glue.DataCollection()
        self._hub = glue.Hub(self._data)
        self.new_tab()
        self.connect()
        self._create_menu()

    def register_to_hub(self):
        self._hub.subscribe(self,
                            ErrorMessage,
                            handler=self._report_error)
    @property
    def tab(self):
        return self._ui.tabWidget

    def new_tab(self):
        """Spawn a new tab page"""
        layout = QGridLayout()
        layout.setSpacing(1)
        layout.setContentsMargins(0,0,0,0)
        widget = QMdiArea()
        widget.setLayout(layout)
        tab = self.tab
        tab.addTab(widget, str(tab.count()+1))
        tab.setCurrentWidget(widget)

    def add_to_current_tab(self, new_widget):
        tab =self.tab
        page = tab.currentWidget()
        print "adding to widget"
        page.addSubWindow(new_widget)

    def connect(self):
        self.register_to_hub()
        self._ui.layerWidget.data_collection = self._data
        self._ui.layerWidget.register_to_hub(self._hub)
        self._data.register_to_hub(self._hub)
        self._ui.actionNew_Tab.triggered.connect(self.new_tab)
        self._ui.actionNew_Tab.setShortcut("Ctrl+T")
        self._ui.actionNew_Tab.setShortcutContext(Qt.ApplicationShortcut)

    def _create_menu(self):
        mbar = self._ui.menubar
        menu = QMenu(mbar)
        menu.setTitle("File")
        menu.addAction(self._ui.actionNew_Tab)
        act = QAction("new scatter", self)
        act.triggered.connect(self._add_scatter)
        menu.addAction(act)
        mbar.addMenu(menu)

    def _report_error(self, message):
        self.statusBar().showMessage(str(message))

    def _add_scatter(self):
        s = glue.qt.ScatterWidget(self._data)
        s.register_to_hub(self._hub)
        self.add_to_current_tab(s)

if __name__ == "__main__":
    main()

