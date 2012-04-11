import sys

from PyQt4.QtGui import QApplication

import cloudviz as cv
from glue.tree_layout import DendrogramLayout
import cloudviz.data_dendro_cpp as cpp
from glue.io import extract_data_fits
from qt_tree_client import QtTreeClient
from qt_subset_browser_client import QtSubsetBrowserClient
from qt_image_client import QtImageClient
from qt_scatter_client import QtScatterClient

def main():
    app = QApplication(sys.argv)

    # read in the data
    data = cpp('../examples/dendro_oph.fits')
    d2 = extract_data_fits('../examples/ha_regrid.fits')
    comp = cv.data.Component(d2['PRIMARY'])
    data.components['ha'] = comp

    hub = cv.Hub()

    #the dendrogram client
    layout = DendrogramLayout(data.tree,
                              data.components['PRIMARY'].data)
    tree_client = QtTreeClient(data, layout=layout)

    # the subset browser client
    subset_client = QtSubsetBrowserClient(data, parent=tree_client)

    # image client
    image_client = QtImageClient(data, parent=tree_client)
    image_client.set_component('PRIMARY')
    
    #scatter client
    scatter_client = QtScatterClient(data, parent=tree_client)
    scatter_client.set_xdata('PRIMARY')
    scatter_client.set_ydata('ha')

    # the subsets
    data.tree.index()
    id = data.tree._index[40].get_subtree_indices()
    s = cv.subset.TreeSubset(data, node_list=id)
    id = data.tree._index[80].get_subtree_indices()
    s2 = cv.subset.TreeSubset(data, node_list=id)
    s2.style.color = 'green'

    # register everything
    data.register_to_hub(hub)
    tree_client.register_to_hub(hub)
    tree_client.refresh()
    subset_client.register_to_hub(hub)
    image_client.register_to_hub(hub)
    scatter_client.register_to_hub(hub)

    s.register()
    s2.register()

    # position and display the guis
    tree_client.show()
    pos = tree_client.pos()
    width = tree_client.width()
    pos.setX(pos.x() + width * 1.1)
    pos.setY(pos.y() - .1 * tree_client.width())    
    subset_client.move(pos)
    pos = tree_client.pos()
    pos.setX(pos.x() - width - image_client.width() * 1.5)

    image_client.move(pos)

    subset_client.show()
    image_client.show()
    scatter_client.show()

    # start event processing
    app.exec_()

if __name__ == "__main__":
    main()
