from __future__ import absolute_import, division, print_function

from glue.core import Data, DataCollection
from glue.external.qt import QtGui

from ..data_combo_helper import (ComponentIDComboHelper, ManualDataComboHelper,
                                 DataCollectionComboHelper)


def _items_as_string(combo):
    items = [combo.itemText(i) for i in range(combo.count())]
    return ":".join(items)


def test_component_id_combo_helper():

    combo = QtGui.QComboBox()

    dc = DataCollection([])

    helper = ComponentIDComboHelper(combo, dc)

    assert _items_as_string(combo) == ""

    data1 = Data(x=[1,2,3], y=[2,3,4], label='data1')

    dc.append(data1)
    helper.append(data1)

    assert _items_as_string(combo) == "x:y"

    data2 = Data(a=[1,2,3], b=['a','b','c'], label='data2')

    dc.append(data2)
    helper.append(data2)

    assert _items_as_string(combo) == "data1:x:y:data2:a:b"

    helper.categorical = False

    assert _items_as_string(combo) == "data1:x:y:data2:a"

    helper.numeric = False

    assert _items_as_string(combo) == "data1:data2"

    helper.categorical = True
    helper.numeric = True

    helper.visible = False
    assert _items_as_string(combo) == "data1:Pixel Axis 0:World 0:x:y:data2:Pixel Axis 0:World 0:a:b"
    helper.visible = True

    dc.remove(data2)

    assert _items_as_string(combo) == "x:y"

    helper.remove(data1)

    assert _items_as_string(combo) == ""


