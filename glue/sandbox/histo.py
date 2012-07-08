import sys

from PyQt4.QtGui import QApplication, QMainWindow

from glue.tests import example_data
from glue.core.data_collection import DataCollection
from glue.core.hub import Hub
from glue.qt.widgets.histogramwidget import HistogramWidget

def main():

    app = QApplication(sys.argv)
    win = QMainWindow()

    data = example_data.simple_image()
    dc = DataCollection([data])
    histo_client = HistogramWidget(dc)

    hub = Hub(dc, histo_client)
    win.setCentralWidget(histo_client)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()