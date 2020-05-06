# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'options_widget.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from glue.viewers.matplotlib.qt.axes_editor import AxesEditorWidget
from glue.viewers.matplotlib.qt.legend_editor import LegendEditorWidget


class Ui_Widget(object):
    def setupUi(self, Widget):
        if not Widget.objectName():
            Widget.setObjectName(u"Widget")
        Widget.resize(269, 418)
        self.gridLayout_5 = QGridLayout(Widget)
        self.gridLayout_5.setSpacing(6)
        self.gridLayout_5.setContentsMargins(11, 11, 11, 11)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridLayout_5.setVerticalSpacing(5)
        self.gridLayout_5.setContentsMargins(5, 5, 5, 5)
        self.tab_widget = QTabWidget(Widget)
        self.tab_widget.setObjectName(u"tab_widget")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.gridLayout_2 = QGridLayout(self.tab)
        self.gridLayout_2.setSpacing(6)
        self.gridLayout_2.setContentsMargins(11, 11, 11, 11)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setHorizontalSpacing(10)
        self.gridLayout_2.setVerticalSpacing(5)
        self.gridLayout_2.setContentsMargins(10, 10, 10, 10)
        self.label_6 = QLabel(self.tab)
        self.label_6.setObjectName(u"label_6")
        font = QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.label_6, 1, 0, 1, 1)

        self.combosel_function = QComboBox(self.tab)
        self.combosel_function.setObjectName(u"combosel_function")

        self.gridLayout_2.addWidget(self.combosel_function, 0, 1, 1, 2)

        self.combosel_x_att = QComboBox(self.tab)
        self.combosel_x_att.setObjectName(u"combosel_x_att")
        self.combosel_x_att.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLength)

        self.gridLayout_2.addWidget(self.combosel_x_att, 2, 1, 1, 2)

        self.label = QLabel(self.tab)
        self.label.setObjectName(u"label")
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)

        self.label_3 = QLabel(self.tab)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setFont(font)
        self.label_3.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.label_3, 2, 0, 1, 1)

        self.bool_normalize = QCheckBox(self.tab)
        self.bool_normalize.setObjectName(u"bool_normalize")

        self.gridLayout_2.addWidget(self.bool_normalize, 4, 1, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 5, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer, 6, 1, 1, 2)

        self.label_7 = QLabel(self.tab)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setFont(font)
        self.label_7.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.label_7, 4, 0, 1, 1)

        self.text_warning = QLabel(self.tab)
        self.text_warning.setObjectName(u"text_warning")
        self.text_warning.setStyleSheet(u"color: rgb(255, 33, 28)")
        self.text_warning.setAlignment(Qt.AlignCenter)
        self.text_warning.setWordWrap(True)

        self.gridLayout_2.addWidget(self.text_warning, 5, 1, 1, 2)

        self.layout_slices = QVBoxLayout()
        self.layout_slices.setSpacing(6)
        self.layout_slices.setObjectName(u"layout_slices")

        self.gridLayout_2.addLayout(self.layout_slices, 6, 0, 1, 3)

        self.combosel_reference_data = QComboBox(self.tab)
        self.combosel_reference_data.setObjectName(u"combosel_reference_data")
        self.combosel_reference_data.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLength)

        self.gridLayout_2.addWidget(self.combosel_reference_data, 1, 1, 1, 2)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer_2, 6, 0, 1, 1)

        self.tab_widget.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.gridLayout = QGridLayout(self.tab_2)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setContentsMargins(11, 11, 11, 11)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setHorizontalSpacing(10)
        self.gridLayout.setVerticalSpacing(5)
        self.gridLayout.setContentsMargins(10, 10, 10, 10)
        self.button_flip_x = QToolButton(self.tab_2)
        self.button_flip_x.setObjectName(u"button_flip_x")
        self.button_flip_x.setStyleSheet(u"padding: 0px")

        self.gridLayout.addWidget(self.button_flip_x, 0, 2, 1, 1)

        self.valuetext_y_min = QLineEdit(self.tab_2)
        self.valuetext_y_min.setObjectName(u"valuetext_y_min")

        self.gridLayout.addWidget(self.valuetext_y_min, 1, 1, 1, 1)

        self.valuetext_x_max = QLineEdit(self.tab_2)
        self.valuetext_x_max.setObjectName(u"valuetext_x_max")

        self.gridLayout.addWidget(self.valuetext_x_max, 0, 3, 1, 1)

        self.valuetext_x_min = QLineEdit(self.tab_2)
        self.valuetext_x_min.setObjectName(u"valuetext_x_min")

        self.gridLayout.addWidget(self.valuetext_x_min, 0, 1, 1, 1)

        self.label_2 = QLabel(self.tab_2)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setFont(font)

        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)

        self.label_5 = QLabel(self.tab_2)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setFont(font)

        self.gridLayout.addWidget(self.label_5, 1, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer, 3, 3, 1, 1)

        self.valuetext_y_max = QLineEdit(self.tab_2)
        self.valuetext_y_max.setObjectName(u"valuetext_y_max")

        self.gridLayout.addWidget(self.valuetext_y_max, 1, 3, 1, 1)

        self.horizontalSpacer_2 = QSpacerItem(40, 5, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_2, 2, 1, 1, 4)

        self.bool_y_log = QToolButton(self.tab_2)
        self.bool_y_log.setObjectName(u"bool_y_log")
        self.bool_y_log.setCheckable(True)

        self.gridLayout.addWidget(self.bool_y_log, 1, 4, 1, 1)

        self.bool_x_log = QToolButton(self.tab_2)
        self.bool_x_log.setObjectName(u"bool_x_log")
        self.bool_x_log.setCheckable(True)

        self.gridLayout.addWidget(self.bool_x_log, 0, 4, 1, 1)

        self.tab_widget.addTab(self.tab_2, "")
        self.bool_x_log.raise_()
        self.valuetext_x_max.raise_()
        self.button_flip_x.raise_()
        self.valuetext_x_min.raise_()
        self.valuetext_y_min.raise_()
        self.valuetext_y_max.raise_()
        self.bool_y_log.raise_()
        self.label_2.raise_()
        self.label_5.raise_()
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.horizontalLayout = QHBoxLayout(self.tab_3)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(5, 5, 5, 5)
        self.axes_editor = AxesEditorWidget(self.tab_3)
        self.axes_editor.setObjectName(u"axes_editor")

        self.horizontalLayout.addWidget(self.axes_editor)

        self.tab_widget.addTab(self.tab_3, "")
        self.tab_4 = QWidget()
        self.tab_4.setObjectName(u"tab_4")
        self.horizontalLayout1 = QHBoxLayout(self.tab_4)
        self.horizontalLayout1.setSpacing(6)
        self.horizontalLayout1.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout1.setObjectName(u"horizontalLayout1")
        self.horizontalLayout1.setContentsMargins(5, 5, 5, 5)
        self.legend_editor = LegendEditorWidget(self.tab_4)
        self.legend_editor.setObjectName(u"legend_editor")

        self.horizontalLayout1.addWidget(self.legend_editor)

        self.tab_widget.addTab(self.tab_4, "")

        self.gridLayout_5.addWidget(self.tab_widget, 9, 2, 1, 1)


        self.retranslateUi(Widget)

        self.tab_widget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(Widget)
    # setupUi

    def retranslateUi(self, Widget):
        Widget.setWindowTitle(QCoreApplication.translate("Widget", u"1D Profile", None))
        self.label_6.setText(QCoreApplication.translate("Widget", u"reference", None))
        self.label.setText(QCoreApplication.translate("Widget", u"function", None))
        self.label_3.setText(QCoreApplication.translate("Widget", u"x axis", None))
        self.bool_normalize.setText("")
        self.label_7.setText(QCoreApplication.translate("Widget", u"normalize", None))
        self.text_warning.setText(QCoreApplication.translate("Widget", u"Warning", None))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab), QCoreApplication.translate("Widget", u"General", None))
        self.button_flip_x.setText(QCoreApplication.translate("Widget", u"\u21c4", None))
        self.label_2.setText(QCoreApplication.translate("Widget", u"x axis", None))
        self.label_5.setText(QCoreApplication.translate("Widget", u"y axis", None))
        self.bool_y_log.setText(QCoreApplication.translate("Widget", u"log", None))
        self.bool_x_log.setText(QCoreApplication.translate("Widget", u"log", None))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab_2), QCoreApplication.translate("Widget", u"Limits", None))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab_3), QCoreApplication.translate("Widget", u"Axes", None))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab_4), QCoreApplication.translate("Widget", u"Legend", None))
    # retranslateUi

