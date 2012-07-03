from glue.util import identity
import glue.qt

""" Visualization Clients """
qt_clients = [glue.qt.ScatterWidget,
              glue.qt.ImageWidget]

"""Functions to use in the link editor to define mapping between
ComponentIDs
"""
link_functions = [identity]


""" Feel free to define or import any extra functions. These will be
visible to the custom component definer """

