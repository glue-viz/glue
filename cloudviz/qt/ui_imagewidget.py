# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'imagewidget.ui'
#
# Created: Sun Mar  4 20:45:22 2012
#      by: PyQt4 UI code generator 4.8.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_ImageWidget(object):
    def setupUi(self, ImageWidget):
        ImageWidget.setObjectName(_fromUtf8("ImageWidget"))
        ImageWidget.resize(737, 619)
        self.horizontalLayout_5 = QtGui.QHBoxLayout(ImageWidget)
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.splitter = QtGui.QSplitter(ImageWidget)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName(_fromUtf8("splitter"))
        self.widget = QtGui.QWidget(self.splitter)
        self.widget.setObjectName(_fromUtf8("widget"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.widget)
        self.verticalLayout_3.setMargin(0)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.mplWidget = MplWidget(self.widget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(200)
        sizePolicy.setVerticalStretch(200)
        sizePolicy.setHeightForWidth(self.mplWidget.sizePolicy().hasHeightForWidth())
        self.mplWidget.setSizePolicy(sizePolicy)
        self.mplWidget.setMinimumSize(QtCore.QSize(400, 400))
        self.mplWidget.setObjectName(_fromUtf8("mplWidget"))
        self.verticalLayout_3.addWidget(self.mplWidget)
        self.imageSlider = QtGui.QSlider(self.widget)
        self.imageSlider.setEnabled(True)
        self.imageSlider.setOrientation(QtCore.Qt.Horizontal)
        self.imageSlider.setInvertedAppearance(False)
        self.imageSlider.setInvertedControls(False)
        self.imageSlider.setTickInterval(0)
        self.imageSlider.setObjectName(_fromUtf8("imageSlider"))
        self.verticalLayout_3.addWidget(self.imageSlider)
        self.widget1 = QtGui.QWidget(self.splitter)
        self.widget1.setObjectName(_fromUtf8("widget1"))
        self.horizontalLayout_4 = QtGui.QHBoxLayout(self.widget1)
        self.horizontalLayout_4.setMargin(0)
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label = QtGui.QLabel(self.widget1)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout.addWidget(self.label)
        self.displayDataCombo = QtGui.QComboBox(self.widget1)
        self.displayDataCombo.setObjectName(_fromUtf8("displayDataCombo"))
        self.horizontalLayout.addWidget(self.displayDataCombo)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.horizontalLayout_4.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label_2 = QtGui.QLabel(self.widget1)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_2.addWidget(self.label_2)
        self.attributeComboBox = QtGui.QComboBox(self.widget1)
        self.attributeComboBox.setObjectName(_fromUtf8("attributeComboBox"))
        self.horizontalLayout_2.addWidget(self.attributeComboBox)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.orientationLabel = QtGui.QLabel(self.widget1)
        self.orientationLabel.setEnabled(True)
        self.orientationLabel.setObjectName(_fromUtf8("orientationLabel"))
        self.horizontalLayout_3.addWidget(self.orientationLabel)
        self.sliceComboBox = QtGui.QComboBox(self.widget1)
        self.sliceComboBox.setObjectName(_fromUtf8("sliceComboBox"))
        self.horizontalLayout_3.addWidget(self.sliceComboBox)
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_5.addWidget(self.splitter)

        self.retranslateUi(ImageWidget)
        QtCore.QMetaObject.connectSlotsByName(ImageWidget)

    def retranslateUi(self, ImageWidget):
        ImageWidget.setWindowTitle(QtGui.QApplication.translate("ImageWidget", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.imageSlider.setToolTip(QtGui.QApplication.translate("ImageWidget", "Adjust the position of the slice plane", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("ImageWidget", "Data", None, QtGui.QApplication.UnicodeUTF8))
        self.displayDataCombo.setToolTip(QtGui.QApplication.translate("ImageWidget", "Set which data set to display", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("ImageWidget", "Attribute", None, QtGui.QApplication.UnicodeUTF8))
        self.attributeComboBox.setToolTip(QtGui.QApplication.translate("ImageWidget", "Set which attribute of the data to display", None, QtGui.QApplication.UnicodeUTF8))
        self.orientationLabel.setText(QtGui.QApplication.translate("ImageWidget", "Slice Orientation", None, QtGui.QApplication.UnicodeUTF8))
        self.sliceComboBox.setToolTip(QtGui.QApplication.translate("ImageWidget", "Set how to slice 3D cubes", None, QtGui.QApplication.UnicodeUTF8))

from mplwidget import MplWidget
