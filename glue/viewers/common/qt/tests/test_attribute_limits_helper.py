import operator

import pytest
import numpy as np

from glue.external.qt import QtGui
from glue.core.data import Data
from glue.core.subset import InequalitySubsetState

from ..attribute_limits_helper import AttributeLimitsHelper

# TEMPORARY
from glue.external.qt import get_qapp
get_qapp()


class TestAttributeLimitsHelper():

    def setup_method(self, method):

        self.attribute_combo = QtGui.QComboBox()
        self.lower_value = QtGui.QLineEdit()
        self.upper_value = QtGui.QLineEdit()
        self.mode_combo = QtGui.QComboBox()
        self.flip_button = QtGui.QToolButton()

        self.helper = AttributeLimitsHelper(self.attribute_combo,
                                            self.lower_value, self.upper_value,
                                            mode_combo=self.mode_combo,
                                            flip_button=self.flip_button)

        self.data = Data(x=np.linspace(-100, 100, 10000),
                         y=np.linspace(2, 3, 10000), label='test_data')

        self.helper.data = self.data

        self.x_id = self.data.visible_components[0]
        self.y_id = self.data.visible_components[1]

    def test_attributes(self):
        assert self.attribute_combo.count() == 2
        assert self.attribute_combo.itemText(0) == 'x'
        assert self.attribute_combo.itemData(0) is self.x_id
        assert self.attribute_combo.itemText(1) == 'y'
        assert self.attribute_combo.itemData(1) is self.y_id

    def test_minmax(self):
        assert self.helper.vlo == -100
        assert self.helper.vhi == +100

    def test_change_attribute(self):
        self.helper.attribute = self.y_id
        assert self.helper.vlo == 2
        assert self.helper.vhi == 3
        self.helper.attribute = self.x_id
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
        self.helper.attribute = self.y_id
        assert self.helper.scale_mode == 'Min/Max'
        self.helper.scale_mode = '99%'
        self.helper.attribute = self.x_id
        assert self.helper.scale_mode == '99.5%'
        self.helper.attribute = self.y_id
        assert self.helper.scale_mode == '99%'

    def test_flip_button(self):

        # Flipping should swap lower and upper value
        self.flip_button.clicked.emit(True)
        assert self.helper.vlo == +100
        assert self.helper.vhi == -100

        # Make sure that values were re-cached when flipping
        self.helper.attribute = self.y_id
        assert self.helper.vlo == 2
        assert self.helper.vhi == 3
        self.helper.attribute = self.x_id
        assert self.helper.vlo == +100
        assert self.helper.vhi == -100

    def test_manual_edit(self):

        # Make sure that values are re-cached when edited manually
        self.helper.mode_combo = 'Custom'
        self.lower_value.setText('-122')
        self.upper_value.setText('234')
        assert self.helper.vlo == -122
        assert self.helper.vhi == 234
        self.helper.attribute = self.y_id
        self.helper.attribute = self.x_id
        assert self.helper.vlo == -122
        assert self.helper.vhi == 234

    def test_subset_mode(self):

        with pytest.raises(ValueError) as exc:
            self.helper.subset_mode = 'data'
        assert exc.value.args[0] == "subset_mode should be set to None when data is not a subset"

        subset_state = InequalitySubsetState(self.x_id, 10, operator.gt)

        self.data.new_subset(subset_state)

        subset = self.data.subsets[0]

        self.helper.data = subset

        with pytest.raises(ValueError) as exc:
            self.helper.subset_mode = None
        assert exc.value.args[0] == "subset_mode should either be 'outline', 'data' when data is a subset"

        self.helper.subset_mode = 'data'
        assert self.helper.vlo == 0
        assert self.helper.vhi == 100
        self.helper.vhi = 56

        self.helper.subset_mode = 'outline'
        assert self.helper.vlo == 0
        assert self.helper.vhi == 2

        self.helper.subset_mode = 'data'
        assert self.helper.vlo == 0
        assert self.helper.vhi == 56
