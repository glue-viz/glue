# Need to use absolute imports here because
from glue.qt.widgets.scatter_widget import ScatterWidget
from glue.qt.widgets.image_widget import ImageWidget
from glue.core.util import identity

""" Visualization Clients """
qt_clients = [ScatterWidget,
              ImageWidget]

"""Functions to use in the link editor to define mapping between
ComponentIDs
"""
link_functions = [identity]


""" Feel free to define or import any extra functions. These will be
visible to the custom component definer """

