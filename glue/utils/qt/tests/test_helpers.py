from qtpy import QtWidgets
from ..helpers import update_combobox


def test_update_combobox():
    combo = QtWidgets.QComboBox()
    update_combobox(combo, [('a', 1), ('b', 2)])
    update_combobox(combo, [('c', 3)])


def test_update_combobox_indexchanged():

    # Regression test for bug that caused currentIndexChanged to not be
    # emitted if the new index happened to be the same as the old one but the
    # label data was different.

    class MyComboBox(QtWidgets.QComboBox):

        def __init__(self, *args, **kwargs):
            self.change_count = 0
            super(MyComboBox, self).__init__(*args, **kwargs)
            self.currentIndexChanged.connect(self.changed)

        def changed(self):
            self.change_count += 1

    combo = MyComboBox()
    update_combobox(combo, [('a', 1), ('b', 2)])
    update_combobox(combo, [('c', 3)])

    assert combo.change_count == 2
    assert combo.currentIndex() == 0

    combo = MyComboBox()
    update_combobox(combo, [('a', 1), ('b', 2)])
    update_combobox(combo, [('a', 1), ('b', 3)])
    update_combobox(combo, [('a', 3), ('b', 1)])

    assert combo.change_count == 3
    assert combo.currentIndex() == 1
