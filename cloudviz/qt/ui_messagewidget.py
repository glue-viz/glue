# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'messagewidget.ui'
#
# Created: Tue Feb 28 13:53:16 2012
#      by: PyQt4 UI code generator 4.8.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_MessageWidget(object):
    def setupUi(self, MessageWidget):
        MessageWidget.setObjectName(_fromUtf8("MessageWidget"))
        MessageWidget.resize(700, 400)
        self.horizontalLayout = QtGui.QHBoxLayout(MessageWidget)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.messageTable = QtGui.QTableWidget(MessageWidget)
        self.messageTable.setObjectName(_fromUtf8("messageTable"))
        self.messageTable.setColumnCount(0)
        self.messageTable.setRowCount(0)
        self.horizontalLayout.addWidget(self.messageTable)

        self.retranslateUi(MessageWidget)
        QtCore.QMetaObject.connectSlotsByName(MessageWidget)

    def retranslateUi(self, MessageWidget):
        MessageWidget.setWindowTitle(QtGui.QApplication.translate("MessageWidget", "Message Widget", None, QtGui.QApplication.UnicodeUTF8))

