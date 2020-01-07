import os
from collections import OrderedDict

from qtpy import QtWidgets
from qtpy.QtCore import Qt

from glue.utils.qt import load_ui, CenteredDialog

__all__ = ['MetadataDialog']


class MetadataDialog(CenteredDialog):
    """
    A dialog to view the metadata in a data object.
    """

    def __init__(self, data, *args, **kwargs):

        super(MetadataDialog, self).__init__(*args, **kwargs)

        self.ui = load_ui('metadata.ui', self, directory=os.path.dirname(__file__))

        self.resize(400, 500)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        self._text = ""
        for name, value in OrderedDict(data.meta).items():
            QtWidgets.QTreeWidgetItem(self.ui.meta_tree.invisibleRootItem(), [name, str(value)])

        if data.label:
            self.setWindowTitle("Metadata for {0}".format(data.label))

        self.ui.label_ndim.setText(str(data.ndim))
        self.ui.label_shape.setText(str(data.shape))

        self.center()
