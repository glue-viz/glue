from __future__ import absolute_import, division, print_function

from glue.external.qt import QtGui
from glue.utils.qt import PyMimeData, GlueItemWidget

# some standard glue mime types
LAYER_MIME_TYPE = 'glue/layer'
LAYERS_MIME_TYPE = 'glue/layers'
INSTANCE_MIME_TYPE = PyMimeData.MIME_TYPE

class GlueMimeListWidget(GlueItemWidget, QtGui.QListWidget):
    SUPPORTED_MIME_TYPE = LAYERS_MIME_TYPE
