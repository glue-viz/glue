from glue.external.qt import QtCore, QtGui, is_pyqt5
from glue.qt.data_collection_model import DataCollectionView
from glue.qt.qtutil import GlueActionButton, get_icon


class Ui_LayerTree(object):

    def setupUi(self, LayerTree):
        font = QtGui.QFont()
        font.setPointSize(11)

        LayerTree.setObjectName("LayerTree")
        LayerTree.resize(241, 282)

        self.layout = QtGui.QVBoxLayout(LayerTree)
        self.layout.setSpacing(2)
        self.layout.setContentsMargins(5, 5, 5, 0)
        self.layout.setObjectName("layout")

        self.layerTree = DataCollectionView(LayerTree)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred,
                                       QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(10)
        sizePolicy.setHeightForWidth(self.layerTree.sizePolicy().hasHeightForWidth())
        self.layerTree.setSizePolicy(sizePolicy)
        self.layerTree.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.layerTree.setObjectName("layerTree")
        self.layout.addWidget(self.layerTree)

        self.button_row = QtGui.QHBoxLayout()
        self.button_row.setSpacing(3)
        self.button_row.setObjectName("button_row")

        self.layerAddButton = QtGui.QPushButton(LayerTree)
        self.layerAddButton.setFont(font)
        icon = get_icon('glue_open')
        self.layerAddButton.setIcon(icon)
        self.layerAddButton.setIconSize(QtCore.QSize(18, 18))
        self.layerAddButton.setDefault(False)
        self.layerAddButton.setObjectName("layerAddButton")

        self.button_row.addWidget(self.layerAddButton)
        self.newSubsetButton = GlueActionButton(LayerTree)
        self.newSubsetButton.setIcon(get_icon('glue_subset'))
        self.newSubsetButton.setIconSize(QtCore.QSize(18, 18))
        self.newSubsetButton.setObjectName("newSubsetButton")
        self.button_row.addWidget(self.newSubsetButton)
        self.newSubsetButton.setFont(font)

        self.layerRemoveButton = QtGui.QPushButton(LayerTree)
        self.layerRemoveButton.setEnabled(False)
        self.layerRemoveButton.setIcon(get_icon('glue_delete'))
        self.layerRemoveButton.setIconSize(QtCore.QSize(18, 18))
        self.layerRemoveButton.setObjectName("layerRemoveButton")
        self.layerRemoveButton.setFont(font)
        self.button_row.addWidget(self.layerRemoveButton)

        self.linkButton = GlueActionButton(LayerTree)
        self.linkButton.setEnabled(True)
        self.linkButton.setIcon(get_icon('glue_link'))
        self.linkButton.setObjectName("linkButton")
        self.linkButton.setIconSize(QtCore.QSize(18, 18))
        self.linkButton.setFont(font)
        self.button_row.addWidget(self.linkButton)

        spacerItem = QtGui.QSpacerItem(20, 20,
                                       QtGui.QSizePolicy.Expanding,
                                       QtGui.QSizePolicy.Minimum)
        self.button_row.addItem(spacerItem)
        self.layout.addLayout(self.button_row)

        self.retranslateUi(LayerTree)
        QtCore.QMetaObject.connectSlotsByName(LayerTree)

    def retranslateUi(self, LayerTree):
        if is_pyqt5():
            LayerTree.setWindowTitle(QtGui.QApplication.translate("LayerTree", "Form", None))
            self.layerAddButton.setToolTip(QtGui.QApplication.translate("LayerTree", "Load a new data set", None))
            self.newSubsetButton.setToolTip(QtGui.QApplication.translate("LayerTree", "Create a new empty subset", None))
            self.layerRemoveButton.setToolTip(QtGui.QApplication.translate("LayerTree", "Delete Layer", None))
            self.linkButton.setToolTip(QtGui.QApplication.translate("LayerTree", "Link data", None))
        else:
            LayerTree.setWindowTitle(QtGui.QApplication.translate("LayerTree", "Form", None, QtGui.QApplication.UnicodeUTF8))
            self.layerAddButton.setToolTip(QtGui.QApplication.translate("LayerTree", "Load a new data set", None, QtGui.QApplication.UnicodeUTF8))
            self.newSubsetButton.setToolTip(QtGui.QApplication.translate("LayerTree", "Create a new empty subset", None, QtGui.QApplication.UnicodeUTF8))
            self.layerRemoveButton.setToolTip(QtGui.QApplication.translate("LayerTree", "Delete Layer", None, QtGui.QApplication.UnicodeUTF8))
            self.linkButton.setToolTip(QtGui.QApplication.translate("LayerTree", "Link data", None, QtGui.QApplication.UnicodeUTF8))
