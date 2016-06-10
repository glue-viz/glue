from __future__ import absolute_import, division, print_function

from qtpy.QtCore import Qt
from glue.core import DataCollection, Data

from ..data_collection_model import DataCollectionModel
from glue.core.qt.mime import LAYERS_MIME_TYPE


class TestDataCollectionModel(object):

    def make_model(self, n_data=1, n_subsets=0):
        dc = DataCollection([Data(x=[1, 2, 3]) for _ in range(n_data)])
        for _ in range(n_subsets):
            dc.new_subset_group()
        return DataCollectionModel(dc)

    def test_row_count_empty_index(self):
        model = self.make_model(1, 0)
        assert model.rowCount() == 2

    def test_row_count_data_row(self):
        model = self.make_model(1, 0)
        assert model.rowCount(model.data_index()) == 1

        model = self.make_model(2, 0)
        assert model.rowCount(model.data_index()) == 2

    def test_row_count_subset_row(self):
        model = self.make_model(1, 0)
        assert model.rowCount(model.subsets_index()) == 0

        model = self.make_model(1, 5)
        assert model.rowCount(model.subsets_index()) == 5

    def test_row_count_single_subset(self):
        model = self.make_model(2, 1)
        assert model.rowCount(model.subsets_index(0)) == 2

    def test_row_count_single_subset(self):
        model = self.make_model(2, 1)
        s = model.subsets_index(0)

        idx = model.index(0, 0, s)
        assert model.rowCount(idx) == 0

        idx = model.index(1, 0, s)
        assert model.rowCount(s) == 2

    def test_invalid_indices(self):
        model = self.make_model(1, 2)

        index = model.index(0, 1)
        assert not index.isValid()

        index = model.index(2, 0)
        assert not index.isValid()

        index = model.index(2, 0, model.index(0, 0))
        assert not index.isValid()

    def test_heading_labels(self):
        model = self.make_model()
        assert model.data(model.data_index(), Qt.DisplayRole) == 'Data'
        assert model.data(model.subsets_index(), Qt.DisplayRole) == 'Subsets'

    def test_dc_labels(self):
        model = self.make_model(1, 2)
        dc = model.data_collection
        dc[0].label = 'test1'
        dc[0].subsets[0].label = 'subset1'
        dc[0].subsets[1].label = 'subset2'

        assert model.data(model.data_index(0), Qt.DisplayRole) == 'test1'
        assert model.data(model.subsets_index(0), Qt.DisplayRole) == 'subset1'
        assert model.data(model.subsets_index(1), Qt.DisplayRole) == 'subset2'
        assert model.data(model.index(0, 0, model.subsets_index(0)),
                          Qt.DisplayRole) == 'subset1 (test1)'

    def test_column_count(self):
        model = self.make_model(1, 2)

        assert model.columnCount(model.data_index()) == 1
        assert model.columnCount(model.data_index(0)) == 1
        assert model.columnCount(model.subsets_index()) == 1
        assert model.columnCount(model.subsets_index(0)) == 1
        assert model.columnCount(model.subsets_index(1)) == 1

    def test_header_data(self):
        model = self.make_model()

        assert model.headerData(0, Qt.Vertical) == ''
        assert model.headerData(0, Qt.Horizontal) == ''

    def test_font_role(self):
        model = self.make_model(1, 2)

        assert model.data(model.data_index(), Qt.FontRole).bold()
        assert model.data(model.subsets_index(), Qt.FontRole).bold()

    def test_drag_flags(self):
        model = self.make_model(1, 2)

        sg = model.subsets_index(0)
        subset = model.index(0, 0, sg)
        assert model.flags(model.data_index(0)) & Qt.ItemIsDragEnabled
        assert model.flags(subset) & Qt.ItemIsDragEnabled

        assert not model.flags(model.data_index()) & Qt.ItemIsDragEnabled
        assert not model.flags(model.subsets_index()) & Qt.ItemIsDragEnabled
        assert not model.flags(sg) & Qt.ItemIsDragEnabled

    def test_selectable_flags(self):
        model = self.make_model(1, 2)

        assert not model.flags(model.data_index()) & Qt.ItemIsSelectable
        assert not model.flags(model.subsets_index()) & Qt.ItemIsSelectable

    def test_layers_mime_type_data(self):
        model = self.make_model(1, 2)
        index = model.data_index(0)

        expected = [model.data_collection[0]]
        assert model.mimeData([index]).data(LAYERS_MIME_TYPE) == expected

    def test_layers_mime_type_multiselection(self):
        model = self.make_model(1, 2)
        idxs = [model.data_index(0),
                model.subsets_index(0),
                model.index(0, 0, model.subsets_index(0))]

        dc = model.data_collection
        expected = [dc[0], dc.subset_groups[0], dc.subset_groups[0].subsets[0]]
        assert model.mimeData(idxs).data(LAYERS_MIME_TYPE) == expected
