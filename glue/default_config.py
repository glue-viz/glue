# Need to use absolute imports here because
import glue

""" Visualization Clients """
qt_clients = [glue.qt.widgets.scatter_widget.ScatterWidget,
              glue.qt.widgets.image_widget.ImageWidget]

"""Functions to use in the link editor to define mapping between
ComponentIDs
"""
link_functions = [glue.util.identity]


""" Feel free to define or import any extra functions. These will be
visible to the custom component definer """

