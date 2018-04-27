from __future__ import absolute_import, division, print_function

import pytest

from qtpy.QtCore import Qt

from ..python_list_model import PythonListModel


class TestListModel(object):

    def test_row_count(self):
        assert PythonListModel([]).rowCount() == 0
        assert PythonListModel([1]).rowCount() == 1
        assert PythonListModel([1, 2]).rowCount() == 2

    def test_data_display(self):
        m = PythonListModel([1, 'a'])
        i = m.index(0)
        assert m.data(i, role=Qt.DisplayRole) == '1'

        i = m.index(1)
        assert m.data(i, role=Qt.DisplayRole) == 'a'

    def test_data_edit(self):
        m = PythonListModel([1, 'a'])
        i = m.index(0)
        assert m.data(i, role=Qt.EditRole) == '1'

        i = m.index(1)
        assert m.data(i, role=Qt.EditRole) == 'a'

    def test_data_user(self):
        m = PythonListModel([1, 'a'])
        i = m.index(0)
        assert m.data(i, role=Qt.UserRole) == 1

        i = m.index(1)
        assert m.data(i, role=Qt.UserRole) == 'a'

    def test_itemget(self):
        m = PythonListModel([1, 'a'])
        assert m[0] == 1
        assert m[1] == 'a'

    def test_itemset(self):
        m = PythonListModel([1, 'a'])
        m[0] = 'b'
        assert m[0] == 'b'

    @pytest.mark.parametrize('items', ([], [1, 2, 3], [1]))
    def test_len(self, items):
        assert len(PythonListModel(items)) == len(items)

    def test_pop(self):
        m = PythonListModel([1, 2, 3])
        assert m.pop() == 3
        assert len(m) == 2
        assert m.pop(0) == 1
        assert len(m) == 1
        assert m[0] == 2

    def test_append(self):
        m = PythonListModel([])
        m.append(2)
        assert m[0] == 2
        m.append(3)
        assert m[1] == 3
        m.pop()
        m.append(4)
        assert m[1] == 4

    def test_extend(self):
        m = PythonListModel([])
        m.extend([2, 3])
        assert m[0] == 2
        assert m[1] == 3

    def test_insert(self):
        m = PythonListModel([1, 2, 3])
        m.insert(1, 5)
        assert m[1] == 5

    def test_iter(self):
        m = PythonListModel([1, 2, 3])
        assert list(m) == [1, 2, 3]
