from __future__ import absolute_import, division, print_function

from mock import MagicMock

from glue.core import Data, DataCollection
from glue.core.component_id import ComponentID
from qtpy import QtWidgets
from glue.utils.qt import combo_as_string

from ..data_combo_helper import (ComponentIDComboHelper, ManualDataComboHelper,
                                 DataCollectionComboHelper)


def test_component_id_combo_helper():

    combo = QtWidgets.QComboBox()

    dc = DataCollection([])

    helper = ComponentIDComboHelper(combo, dc)

    assert combo_as_string(combo) == ""

    data1 = Data(x=[1, 2, 3], y=[2, 3, 4], label='data1')

    dc.append(data1)
    helper.append_data(data1)

    assert combo_as_string(combo) == "x:y"

    data2 = Data(a=[1, 2, 3], b=['a', 'b', 'c'], label='data2')

    dc.append(data2)
    helper.append_data(data2)

    assert combo_as_string(combo) == "data1:x:y:data2:a:b"

    helper.categorical = False

    assert combo_as_string(combo) == "data1:x:y:data2:a"

    helper.numeric = False

    assert combo_as_string(combo) == "data1:data2"

    helper.categorical = True
    helper.numeric = True

    helper.visible = False
    assert combo_as_string(combo) == "data1:x:Pixel Axis 0 [x]:World 0:y:data2:a:Pixel Axis 0 [x]:World 0:b"
    helper.visible = True

    dc.remove(data2)

    assert combo_as_string(combo) == "x:y"

    # TODO: check that renaming a component updates the combo
    # data1.id['x'].label = 'z'
    # assert combo_as_string(combo) == "z:y"

    helper.remove_data(data1)

    assert combo_as_string(combo) == ""


def test_component_id_combo_helper_init():

    # Regression test to make sure that the numeric and categorical options
    # in the __init__ are taken into account properly

    combo = QtWidgets.QComboBox()

    dc = DataCollection([])

    data = Data(a=[1,2,3], b=['a','b','c'], label='data2')
    dc.append(data)

    helper = ComponentIDComboHelper(combo, dc)
    helper.append_data(data)
    assert combo_as_string(combo) == "a:b"

    helper = ComponentIDComboHelper(combo, dc, numeric=False)
    helper.append_data(data)
    assert combo_as_string(combo) == "b"

    helper = ComponentIDComboHelper(combo, dc, categorical=False)
    helper.append_data(data)
    assert combo_as_string(combo) == "a"

    helper = ComponentIDComboHelper(combo, dc, numeric=False, categorical=False)
    helper.append_data(data)
    assert combo_as_string(combo) == ""


def test_component_id_combo_helper_replaced():

    # Make sure that when components are replaced, the equivalent combo index
    # remains selected and an event is broadcast so that any attached callback
    # properties can be sure to pull the latest text/userData.

    callback = MagicMock()

    combo = QtWidgets.QComboBox()
    combo.currentIndexChanged.connect(callback)

    dc = DataCollection([])

    helper = ComponentIDComboHelper(combo, dc)

    assert combo_as_string(combo) == ""

    data1 = Data(x=[1, 2, 3], y=[2, 3, 4], label='data1')

    callback.reset_mock()

    dc.append(data1)
    helper.append_data(data1)

    callback.assert_called_once_with(0)
    callback.reset_mock()

    assert combo_as_string(combo) == "x:y"

    new_id = ComponentID(label='new')

    data1.update_id(data1.id['x'], new_id)

    callback.assert_called_once_with(0)
    callback.reset_mock()

    assert combo_as_string(combo) == "new:y"


def test_manual_data_combo_helper():

    combo = QtWidgets.QComboBox()

    dc = DataCollection([])

    helper = ManualDataComboHelper(combo, dc)

    data1 = Data(x=[1,2,3], y=[2,3,4], label='data1')

    dc.append(data1)

    assert combo_as_string(combo) == ""

    helper.append_data(data1)

    assert combo_as_string(combo) == "data1"

    data1.label = 'mydata1'
    assert combo_as_string(combo) == "mydata1"

    dc.remove(data1)

    assert combo_as_string(combo) == ""


def test_data_collection_combo_helper():

    combo = QtWidgets.QComboBox()

    dc = DataCollection([])

    helper = DataCollectionComboHelper(combo, dc)

    data1 = Data(x=[1,2,3], y=[2,3,4], label='data1')

    dc.append(data1)

    assert combo_as_string(combo) == "data1"

    data1.label = 'mydata1'
    assert combo_as_string(combo) == "mydata1"

    dc.remove(data1)

    assert combo_as_string(combo) == ""
