from PyQt4.QtCore import Qt, SIGNAL
from PyQt4.QtGui import *

from ui_glue_application import Ui_GlueApplication
from glue_viz_loader_widget import GlueVizLoaderWidget
import glue

class GlueApplication(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        self._ui = Ui_GlueApplication()
        self._ui.setupUi(self)
        self._ui.layerWidget.set_checkable(False)
        self._data = glue.DataCollection()
        self._hub = glue.Hub(self._data)

        self.new_tab()
        self.connect()

    def new_tab(self):
        layout = QGridLayout()
        layout.setSpacing(1)
        layout.setContentsMargins(0,0,0,0)
        widget = QWidget()
        widget.setLayout(layout)
        factory = GlueVizLoaderWidget.wrapper_factory
        layout.addWidget(factory(self), 0,0,1,1)
        layout.addWidget(factory(self), 0,1,1,1)
        self._ui.tabWidget.addTab(widget, "new Tab")
        self._ui.tabWidget.setCurrentWidget(widget)

    def connect(self):
        self._ui.layerWidget.data_collection = self._data
        self._ui.layerWidget.register_to_hub(self._hub)
        self._data.register_to_hub(self._hub)
        self._ui.actionNew_Tab.triggered.connect(self.new_tab)

if __name__ == "__main__":
    main()

