# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'subsetlinkerdialog.ui'
#
# Created: Fri Feb 17 14:40:39 2012
#      by: PyQt4 UI code generator 4.8.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_SubsetLinkerDialog(object):
    def setupUi(self, SubsetLinkerDialog):
        SubsetLinkerDialog.setObjectName(_fromUtf8("SubsetLinkerDialog"))
        SubsetLinkerDialog.setWindowModality(QtCore.Qt.WindowModal)
        SubsetLinkerDialog.resize(319, 339)
        SubsetLinkerDialog.setModal(True)
        self.verticalLayout_2 = QtGui.QVBoxLayout(SubsetLinkerDialog)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.label = QtGui.QLabel(SubsetLinkerDialog)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout.addWidget(self.label)
        self.layerTree = QtGui.QTreeWidget(SubsetLinkerDialog)
        self.layerTree.setObjectName(_fromUtf8("layerTree"))
        self.verticalLayout.addWidget(self.layerTree)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.cancelButton = QtGui.QPushButton(SubsetLinkerDialog)
        self.cancelButton.setObjectName(_fromUtf8("cancelButton"))
        self.horizontalLayout.addWidget(self.cancelButton)
        self.okButton = QtGui.QPushButton(SubsetLinkerDialog)
        self.okButton.setEnabled(False)
        self.okButton.setDefault(False)
        self.okButton.setFlat(False)
        self.okButton.setObjectName(_fromUtf8("okButton"))
        self.horizontalLayout.addWidget(self.okButton)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.retranslateUi(SubsetLinkerDialog)
        QtCore.QMetaObject.connectSlotsByName(SubsetLinkerDialog)
        SubsetLinkerDialog.setTabOrder(self.layerTree, self.cancelButton)
        SubsetLinkerDialog.setTabOrder(self.cancelButton, self.okButton)

    def retranslateUi(self, SubsetLinkerDialog):
        SubsetLinkerDialog.setWindowTitle(QtGui.QApplication.translate("SubsetLinkerDialog", "Subset Linker", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("SubsetLinkerDialog", "Select two or more subsets to link", None, QtGui.QApplication.UnicodeUTF8))
        self.layerTree.headerItem().setText(0, QtGui.QApplication.translate("SubsetLinkerDialog", "Layer", None, QtGui.QApplication.UnicodeUTF8))
        self.cancelButton.setText(QtGui.QApplication.translate("SubsetLinkerDialog", "Cancel", None, QtGui.QApplication.UnicodeUTF8))
        self.okButton.setText(QtGui.QApplication.translate("SubsetLinkerDialog", "OK", None, QtGui.QApplication.UnicodeUTF8))

