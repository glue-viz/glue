from __future__ import absolute_import, division, print_function

from glue.core import Data, DataCollection
from qtpy import QtWidgets

from ..data_combo_helper import (ComponentIDComboHelper, ManualDataComboHelper,
                                 DataCollectionComboHelper)


def _items_as_string(combo):
    items = [combo.itemText(i) for i in range(combo.count())]
    return ":".join(items)


def test_component_id_combo_helper():

    combo = QtWidgets.QComboBox()

    dc = DataCollection([])

    helper = ComponentIDComboHelper(combo, dc)

    assert _items_as_string(combo) == ""

    data1 = Data(x=[1,2,3], y=[2,3,4], label='data1')

    dc.append(data1)
    helper.append_data(data1)

    assert _items_as_string(combo) == "x:y"

    data2 = Data(a=[1,2,3], b=['a','b','c'], label='data2')

    dc.append(data2)
    helper.append_data(data2)

    assert _items_as_string(combo) == "data1:x:y:data2:a:b"

    helper.categorical = False

    assert _items_as_string(combo) == "data1:x:y:data2:a"

    helper.numeric = False

    assert _items_as_string(combo) == "data1:data2"

    helper.categorical = True
    helper.numeric = True

    helper.visible = False
    assert _items_as_string(combo) == "data1:Pixel Axis 0 [x]:World 0:x:y:data2:Pixel Axis 0 [x]:World 0:a:b"
    helper.visible = True

    dc.remove(data2)

    assert _items_as_string(combo) == "x:y"

    # TODO: check that renaming a component updates the combo
    # data1.id['x'].label = 'z'
    # assert _items_as_string(combo) == "z:y"

    helper.remove_data(data1)

    assert _items_as_string(combo) == ""

def test_component_id_combo_helper_init():

    # Regression test to make sure that the numeric and categorical options
    # in the __init__ are taken into account properly

    combo = QtWidgets.QComboBox()

    dc = DataCollection([])

    data = Data(a=[1,2,3], b=['a','b','c'], label='data2')
    dc.append(data)

    helper = ComponentIDComboHelper(combo, dc)
    helper.append_data(data)
    assert _items_as_string(combo) == "a:b"

    helper = ComponentIDComboHelper(combo, dc, numeric=False)
    helper.append_data(data)
    assert _items_as_string(combo) == "b"

    helper = ComponentIDComboHelper(combo, dc, categorical=False)
    helper.append_data(data)
    assert _items_as_string(combo) == "a"

    helper = ComponentIDComboHelper(combo, dc, numeric=False, categorical=False)
    helper.append_data(data)
    assert _items_as_string(combo) == ""


def test_manual_data_combo_helper():

    combo = QtWidgets.QComboBox()

    dc = DataCollection([])

    helper = ManualDataComboHelper(combo, dc)

    data1 = Data(x=[1,2,3], y=[2,3,4], label='data1')

    dc.append(data1)

    assert _items_as_string(combo) == ""

    helper.append_data(data1)

    assert _items_as_string(combo) == "data1"

    data1.label = 'mydata1'
    assert _items_as_string(combo) == "mydata1"

    dc.remove(data1)

    assert _items_as_string(combo) == ""


def test_data_collection_combo_helper():

    combo = QtWidgets.QComboBox()

    dc = DataCollection([])

    helper = DataCollectionComboHelper(combo, dc)

    data1 = Data(x=[1,2,3], y=[2,3,4], label='data1')

    dc.append(data1)

    assert _items_as_string(combo) == "data1"

    data1.label = 'mydata1'
    assert _items_as_string(combo) == "mydata1"

    dc.remove(data1)

    assert _items_as_string(combo) == ""
