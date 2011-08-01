import sys

from PyQt4.QtGui import QApplication

import cloudviz as cv
from qt_subset_browser_client import QtSubsetBrowserClient
from qt_scatter_client import QtScatterClient

def main():
    app = QApplication(sys.argv)

    # read in the data
    data = cv.data.TabularData()
    data.read_data('../examples/oph_c2d_yso_catalog.tbl')
    
    hub = cv.Hub()

    # the subset browser client
    subset_client = QtSubsetBrowserClient(data)

    #scatter clients
    scatter_client = QtScatterClient(data, raster=False)
    scatter_client.set_xdata('ra')
    scatter_client.set_ydata('dec')

    scatter_client_2 = QtScatterClient(data, raster=False)
    scatter_client_2.set_xdata('J_flux_c')
    scatter_client_2.set_ydata('H_flux_c')

    # the subsets
    s = cv.subset.ElementSubset(data)
    s2 = cv.subset.ElementSubset(data)
    s2.style['color'] = 'green'

    # register everything
    data.register_to_hub(hub)
    subset_client.register_to_hub(hub)
    scatter_client.register_to_hub(hub)
    scatter_client_2.register_to_hub(hub)

    s.register()
    s2.register()

    #display
    subset_client.show()
    scatter_client.show()
    scatter_client_2.show()

    # start event processing
    app.exec_()

if __name__ == "__main__":
    main()
