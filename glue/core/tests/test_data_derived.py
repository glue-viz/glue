import pytest

import numpy as np
from numpy.testing import assert_equal

from glue.core.hub import HubListener
from glue.core.message import NumericalDataChangedMessage
from glue.core.data import Data
from glue.core.data_collection import DataCollection
from glue.core.data_derived import IndexedData
from glue.core.coordinates import AffineCoordinates


class TestIndexedData:

    def setup_class(self):

        x = +np.arange(2520).reshape((3, 4, 5, 6, 7))
        y = -np.arange(2520).reshape((3, 4, 5, 6, 7))

        self.data = Data(x=x, y=y, label='Test data')
        self.x_id, self.y_id = self.data.main_components

        matrix = np.random.random((6, 6)) - 0.5
        matrix[-1] = [0, 0, 0, 0, 0, 1]
        self.data_with_coords = Data(x=x, y=y, label='Test data',
                                     coords=AffineCoordinates(matrix=matrix))

        self.subset_state = self.x_id >= 1200

    def test_identity(self):

        # In this test, we don't actually slice any dimensions

        derived = IndexedData(self.data, (None,) * 5)

        assert derived.label == 'Test data[:,:,:,:,:]'
        assert derived.shape == self.data.shape
        assert [str(c) for c in derived.main_components] == [str(c) for c in self.data.main_components]
        assert derived.get_kind(self.x_id) == self.data.get_kind(self.x_id)

        for view in [None, (1, slice(None), slice(None), slice(1, 4), slice(0, 7, 2))]:

            assert_equal(derived.get_data(self.x_id, view=view),
                         self.data.get_data(self.x_id, view=view))

            assert_equal(derived.get_mask(self.subset_state, view=view),
                         self.data.get_mask(self.subset_state, view=view))

        bounds = [2, (-5, 5, 10), 3, 4, (-3, 3, 10)]
        assert_equal(derived.compute_fixed_resolution_buffer(bounds=bounds, target_cid=self.x_id),
                     self.data.compute_fixed_resolution_buffer(bounds=bounds, target_cid=self.x_id))

        assert_equal(derived.compute_statistic('mean', self.x_id),
                     self.data.compute_statistic('mean', self.x_id))

        assert_equal(derived.compute_statistic('mean', self.x_id, axis=2),
                     self.data.compute_statistic('mean', self.x_id, axis=2))

        assert_equal(derived.compute_statistic('mean', self.x_id, subset_state=self.subset_state),
                     self.data.compute_statistic('mean', self.x_id, subset_state=self.subset_state))

        assert_equal(derived.compute_histogram([self.x_id], range=[(0, 1000)], bins=[30]),
                     self.data.compute_histogram([self.x_id], range=[(0, 1000)], bins=[30]))

        assert_equal(derived.compute_histogram([self.x_id], range=[(0, 1000)], bins=[30], subset_state=self.subset_state),
                     self.data.compute_histogram([self.x_id], range=[(0, 1000)], bins=[30], subset_state=self.subset_state))

    def test_indexed(self):

        # Here we slice two of the dimensions and then compare the results to a
        # manually sliced dataset.

        derived = IndexedData(self.data, (None, 2, None, 4, None))
        manual = Data()
        manual.add_component(self.data[self.x_id][:, 2, :, 4, :], label=self.x_id)
        manual.add_component(self.data[self.y_id][:, 2, :, 4, :], label=self.y_id)

        assert derived.label == 'Test data[:,2,:,4,:]'
        assert derived.shape == manual.shape
        assert [str(c) for c in derived.main_components] == [str(c) for c in manual.main_components]
        assert derived.get_kind(self.x_id) == manual.get_kind(self.x_id)

        for view in [None, (1, slice(None), slice(1, 4))]:

            assert_equal(derived.get_data(self.x_id, view=view),
                         manual.get_data(self.x_id, view=view))

            assert_equal(derived.get_mask(self.subset_state, view=view),
                         manual.get_mask(self.subset_state, view=view))

        bounds = [2, (-5, 5, 10), (-3, 3, 10)]
        assert_equal(derived.compute_fixed_resolution_buffer(bounds=bounds, target_cid=self.x_id),
                     manual.compute_fixed_resolution_buffer(bounds=bounds, target_cid=self.x_id))

        assert_equal(derived.compute_statistic('mean', self.x_id),
                     manual.compute_statistic('mean', self.x_id))

        assert_equal(derived.compute_statistic('mean', self.x_id, axis=2),
                     manual.compute_statistic('mean', self.x_id, axis=2))

        assert_equal(derived.compute_statistic('mean', self.x_id, subset_state=self.subset_state),
                     manual.compute_statistic('mean', self.x_id, subset_state=self.subset_state))

        assert_equal(derived.compute_histogram([self.x_id], range=[(0, 1000)], bins=[30]),
                     manual.compute_histogram([self.x_id], range=[(0, 1000)], bins=[30]))

        assert_equal(derived.compute_histogram([self.x_id], range=[(0, 1000)], bins=[30], subset_state=self.subset_state),
                     manual.compute_histogram([self.x_id], range=[(0, 1000)], bins=[30], subset_state=self.subset_state))

    def test_numerical_values_changed(self):

        # Here we slice two of the dimensions and then compare the results to a
        # manually sliced dataset.

        derived = IndexedData(self.data, (None, 2, None, 4, None))
        data_collection = DataCollection([self.data, derived])

        class CustomListener(HubListener):

            def __init__(self, hub):
                self.received = 0
                hub.subscribe(self, NumericalDataChangedMessage,
                              handler=self.receive_message)

            def receive_message(self, message):
                self.received += 1

        listener = CustomListener(data_collection.hub)

        assert listener.received == 0

        derived.indices = (None, 3, None, 5, None)

        assert listener.received == 1

        derived.indices = (None, 3, None, 5, None)

        assert listener.received == 1

    def test_invalid_indices_init(self):
        with pytest.raises(ValueError) as exc:
            derived = IndexedData(self.data, (2, None, 4, None))
        assert exc.value.args[0] == "The 'indices' tuple should have length 5"

    def test_invalid_indices_changed(self):

        derived = IndexedData(self.data, (None, 2, None, 4, None))

        with pytest.raises(TypeError) as exc:
            derived.indices = (2, None, 4, None, 5)
        assert exc.value.args[0] == "Can't change where the ``None`` values are in indices"

        with pytest.raises(ValueError) as exc:
            derived.indices = (2, None, 4, None)
        assert exc.value.args[0] == "The 'indices' tuple should have length 5"

    def test_pixel_component_ids(self):
        derived = IndexedData(self.data, (None, 2, None, 4, None))
        assert_equal(derived.get_data(derived.pixel_component_ids[1]),
                     self.data.get_data(self.data.pixel_component_ids[2])[:, 2, :, 4, :])

    def test_world_component_ids(self):

        derived = IndexedData(self.data, (None, 2, None, 4, None))
        assert derived.world_component_ids == []

        derived_with_coords = IndexedData(self.data_with_coords, (None, 2, None, 4, None))
        assert_equal(derived_with_coords.get_data(derived_with_coords.world_component_ids[0]),
                     self.data_with_coords.get_data(self.data_with_coords.world_component_ids[0])[:, 2, :, 4, :])
        assert_equal(derived_with_coords.get_data(derived_with_coords.world_component_ids[1]),
                     self.data_with_coords.get_data(self.data_with_coords.world_component_ids[2])[:, 2, :, 4, :])
        assert_equal(derived_with_coords.get_data(derived_with_coords.world_component_ids[2]),
                     self.data_with_coords.get_data(self.data_with_coords.world_component_ids[4])[:, 2, :, 4, :])
