from collections import OrderedDict

from unittest.mock import MagicMock
import pytest
import numpy as np
from numpy.testing import assert_equal
from glue.core import DataCollection, Data

from ..subset_mask import SubsetMaskImporter, SubsetMaskExporter


class MySubsetMaskImporter(SubsetMaskImporter):

    filename = None
    reader = None

    def get_filename_and_reader(self):
        return self.filename, self.reader


class MySubsetMaskExporter(SubsetMaskExporter):

    filename = None
    writer = None

    def get_filename_and_writer(self):
        return self.filename, self.writer


class TestImporter():

    def setup_method(self, method):
        self.importer = MySubsetMaskImporter()
        self.importer.filename = 'test-filename'
        self.importer.reader = MagicMock()
        self.data = Data(x=[1, 2, 3])
        self.data_collection = DataCollection([self.data])

    def test_single_valid(self):
        self.importer.reader.return_value = OrderedDict([('subset 1', np.array([0, 1, 0]))])
        self.importer.run(self.data, self.data_collection)
        assert len(self.data_collection.subset_groups) == 1
        assert_equal(self.data.subsets[0].to_mask(), [0, 1, 0])

    def test_multiple_valid(self):
        self.importer.reader.return_value = OrderedDict([('subset 1', np.array([0, 1, 0])),
                                                         ('subset 2', np.array([1, 1, 0]))])
        self.importer.run(self.data, self.data_collection)
        assert len(self.data_collection.subset_groups) == 2
        assert_equal(self.data.subsets[0].to_mask(), [0, 1, 0])
        assert_equal(self.data.subsets[1].to_mask(), [1, 1, 0])

    def test_missing_masks(self):
        self.importer.reader.return_value = OrderedDict()
        with pytest.raises(ValueError) as exc:
            self.importer.run(self.data, self.data_collection)
        assert exc.value.args[0] == "No subset masks were returned"

    def test_single_invalid_shape(self):
        self.importer.reader.return_value = OrderedDict([('subset 1', np.array([0, 1, 0, 1]))])
        with pytest.raises(ValueError) as exc:
            self.importer.run(self.data, self.data_collection)
        assert exc.value.args[0].replace('L', '') == "Mask shape (4,) does not match data shape (3,)"

    def test_multiple_inconsistent_shapes(self):
        self.importer.reader.return_value = OrderedDict([('subset 1', np.array([0, 1, 0])),
                                                         ('subset 2', np.array([0, 1, 0, 1]))])
        with pytest.raises(ValueError) as exc:
            self.importer.run(self.data, self.data_collection)
        assert exc.value.args[0] == "Not all subsets have the same shape"

    def test_subset_single(self):
        self.importer.reader.return_value = OrderedDict([('subset 1', np.array([0, 1, 0]))])
        subset = self.data.new_subset()
        assert_equal(self.data.subsets[0].to_mask(), [0, 0, 0])
        self.importer.run(subset, self.data_collection)
        assert_equal(self.data.subsets[0].to_mask(), [0, 1, 0])

    def test_subset_multiple(self):
        self.importer.reader.return_value = OrderedDict([('subset 1', np.array([0, 1, 0])),
                                                         ('subset 2', np.array([1, 1, 0]))])
        subset = self.data.new_subset()
        with pytest.raises(ValueError) as exc:
            self.importer.run(subset, self.data_collection)
        assert exc.value.args[0] == 'Can only read in a single subset when importing into a subset'


class TestExporter():

    def setup_method(self, method):
        self.exporter = MySubsetMaskExporter()
        self.exporter.filename = 'test-filename'
        self.exporter.writer = MagicMock()
        self.data = Data(x=[1, 2, 3])
        self.data_collection = DataCollection([self.data])

    def test_no_subsets(self):
        with pytest.raises(ValueError) as exc:
            self.exporter.run(self.data)
        assert exc.value.args[0] == 'Data has no subsets'

    def test_multiple_valid(self):
        self.subset1 = self.data_collection.new_subset_group(subset_state=self.data.id['x'] >= 2,
                                                             label='subset a')
        self.subset2 = self.data_collection.new_subset_group(subset_state=self.data.id['x'] >= 3,
                                                             label='subset b')
        self.exporter.run(self.data)
        assert self.exporter.writer.call_count == 1
        assert self.exporter.writer.call_args[0][0] == 'test-filename'
        masks = self.exporter.writer.call_args[0][1]
        assert len(masks) == 2
        assert_equal(masks['subset a'], [0, 1, 1])
        assert_equal(masks['subset b'], [0, 0, 1])

    def test_single_subset_valid(self):
        self.subset = self.data_collection.new_subset_group(subset_state=self.data.id['x'] >= 2,
                                                            label='subset a')
        self.exporter.run(self.data.subsets[0])
        assert self.exporter.writer.call_count == 1
        assert self.exporter.writer.call_args[0][0] == 'test-filename'
        masks = self.exporter.writer.call_args[0][1]
        assert len(masks) == 1
        assert_equal(masks['subset a'], [0, 1, 1])
