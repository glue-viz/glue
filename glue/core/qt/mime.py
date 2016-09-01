from __future__ import absolute_import, division, print_function

from qtpy import QtWidgets
from glue.utils.qt import PyMimeData, GlueItemWidget

# some standard glue mime types
LAYER_MIME_TYPE = 'glue/layer'
LAYERS_MIME_TYPE = 'glue/layers'
INSTANCE_MIME_TYPE = PyMimeData.MIME_TYPE

class GlueMimeListWidget(GlueItemWidget, QtWidgets.QListWidget):
    SUPPORTED_MIME_TYPE = LAYERS_MIME_TYPE
