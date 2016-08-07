import operator

import pytest
import numpy as np

from qtpy import QtWidgets
from glue.core import Data, DataCollection
from glue.core.subset import InequalitySubsetState
from glue.core.qt.data_combo_helper import ComponentIDComboHelper

from ..attribute_limits_helper import AttributeLimitsHelper

# TEMPORARY
from glue.utils.qt import get_qapp
get_qapp()


class TestAttributeLimitsHelper():

    def setup_method(self, method):

        self.attribute_combo = QtWidgets.QComboBox()
        self.lower_value = QtWidgets.QLineEdit()
        self.upper_value = QtWidgets.QLineEdit()
        self.mode_combo = QtWidgets.QComboBox()
        self.flip_button = QtWidgets.QToolButton()

        self.log_button = QtWidgets.QToolButton()
        self.log_button.setCheckable(True)

        self.data = Data(x=np.linspace(-100, 100, 10000),
                         y=np.linspace(2, 3, 10000), label='test_data')

        self.data_collection = DataCollection([self.data])

        self.helper = AttributeLimitsHelper(self.attribute_combo,
                                            self.lower_value, self.upper_value,
                                            mode_combo=self.mode_combo,
                                            flip_button=self.flip_button,
                                            log_button=self.log_button)

        self.component_helper = ComponentIDComboHelper(self.attribute_combo, self.data_collection)

        self.component_helper.append_data(self.data)

        self.x_id = self.data.visible_components[0]
        self.y_id = self.data.visible_components[1]

    def test_attributes(self):
        assert self.attribute_combo.count() == 2
        assert self.attribute_combo.itemText(0) == 'x'
        assert self.attribute_combo.itemData(0)[0] is self.x_id
        assert self.attribute_combo.itemData(0)[1] is self.data
        assert self.attribute_combo.itemText(1) == 'y'
        assert self.attribute_combo.itemData(1)[0] is self.y_id
        assert self.attribute_combo.itemData(1)[1] is self.data

    def test_minmax(self):
        assert self.helper.vlo == -100
        assert self.helper.vhi == +100

    def test_change_attribute(self):
        self.attribute_combo.setCurrentIndex(1)
        assert self.helper.vlo == 2
        assert self.helper.vhi == 3
        self.attribute_combo.setCurrentIndex(0)
        assert self.helper.vlo == -100
        assert self.helper.vhi == +100

    def test_change_scale_mode(self):

        # Changing scale mode updates the limits
        self.helper.scale_mode = '99.5%'
        assert self.helper.vlo == -99.5
        assert self.helper.vhi == +99.5
        self.helper.scale_mode = '99%'
        assert self.helper.vlo == -99
        assert self.helper.vhi == +99
        self.helper.scale_mode = '90%'
        assert self.helper.vlo == -90
        assert self.helper.vhi == +90

        # When switching to custom, the last limits are retained
        self.helper.scale_mode = 'Custom'
        assert self.helper.vlo == -90
        assert self.helper.vhi == +90

    def test_scale_mode_cached(self):
        # Make sure that if we change scale and change attribute, the scale
        # modes are cached on a per-attribute basis.
        self.helper.scale_mode = '99.5%'
        self.attribute_combo.setCurrentIndex(1)
        assert self.helper.scale_mode == 'Min/Max'
        self.helper.scale_mode = '99%'
        self.attribute_combo.setCurrentIndex(0)
        assert self.helper.scale_mode == '99.5%'
        self.attribute_combo.setCurrentIndex(1)
        assert self.helper.scale_mode == '99%'

    def test_flip_button(self):

        # Flipping should swap lower and upper value
        try:
            self.flip_button.clicked.emit(True)
        except TypeError:  # PySide
            self.flip_button.clicked.emit()

        assert self.helper.vlo == +100
        assert self.helper.vhi == -100

        # Make sure that values were re-cached when flipping
        self.attribute_combo.setCurrentIndex(1)
        assert self.helper.vlo == 2
        assert self.helper.vhi == 3
        self.attribute_combo.setCurrentIndex(0)
        assert self.helper.vlo == +100
        assert self.helper.vhi == -100

    def test_manual_edit(self):

        # Make sure that values are re-cached when edited manually
        self.helper.scale_mode = 'Custom'
        self.lower_value.setText('-122')
        self.upper_value.setText('234')
        self.helper.vlog = True
        assert self.helper.vlo == -122
        assert self.helper.vhi == 234
        assert self.helper.vlog
        self.attribute_combo.setCurrentIndex(1)
        assert self.helper.vlo == 2
        assert self.helper.vhi == 3
        assert not self.helper.vlog
        self.attribute_combo.setCurrentIndex(0)
        assert self.helper.vlo == -122
        assert self.helper.vhi == 234
        assert self.helper.vlog
