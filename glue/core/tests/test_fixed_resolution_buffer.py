import pytest

import numpy as np
from numpy.testing import assert_equal

from glue.core import Data, DataCollection
from glue.core.link_helpers import LinkSame


ARRAY = np.arange(3024).reshape((6, 7, 8, 9)).astype(float)


class TestFixedResolutionBuffer():

    def setup_method(self, method):

        self.data_collection = DataCollection()

        # The reference dataset. Shape is (6, 7, 8, 9).
        self.data1 = Data(x=ARRAY)
        self.data_collection.append(self.data1)

        # A dataset with the same shape but not linked. Shape is (6, 7, 8, 9).
        self.data2 = Data(x=ARRAY)
        self.data_collection.append(self.data2)

        # A dataset with the same number of dimensions but in a different
        # order, linked to the first. Shape is (9, 7, 6, 8).
        self.data3 = Data(x=np.moveaxis(ARRAY, (3, 1, 0, 2), (0, 1, 2, 3)))
        self.data_collection.append(self.data3)
        self.data_collection.add_link(LinkSame(self.data1.pixel_component_ids[0],
                                               self.data3.pixel_component_ids[2]))
        self.data_collection.add_link(LinkSame(self.data1.pixel_component_ids[1],
                                               self.data3.pixel_component_ids[1]))
        self.data_collection.add_link(LinkSame(self.data1.pixel_component_ids[2],
                                               self.data3.pixel_component_ids[3]))
        self.data_collection.add_link(LinkSame(self.data1.pixel_component_ids[3],
                                               self.data3.pixel_component_ids[0]))

        # A dataset with fewer dimensions, linked to the first one. Shape is
        # (8, 7, 6)
        self.data4 = Data(x=ARRAY[:, :, :, 0].transpose())
        self.data_collection.append(self.data4)
        self.data_collection.add_link(LinkSame(self.data1.pixel_component_ids[0],
                                               self.data4.pixel_component_ids[2]))
        self.data_collection.add_link(LinkSame(self.data1.pixel_component_ids[1],
                                               self.data4.pixel_component_ids[1]))
        self.data_collection.add_link(LinkSame(self.data1.pixel_component_ids[2],
                                               self.data4.pixel_component_ids[0]))

        # A dataset with even fewer dimensions, linked to the first one. Shape
        # is (8, 6)
        self.data5 = Data(x=ARRAY[:, 0, :, 0].transpose())
        self.data_collection.append(self.data5)
        self.data_collection.add_link(LinkSame(self.data1.pixel_component_ids[0],
                                               self.data5.pixel_component_ids[1]))
        self.data_collection.add_link(LinkSame(self.data1.pixel_component_ids[2],
                                               self.data5.pixel_component_ids[0]))

        # A dataset that is not on the same pixel grid and requires reprojection
        # self.data6 = Data()
        # self.data6.coords = SimpleCoordinates()
        # self.array_nonaligned = np.arange(60).reshape((5, 3, 4))
        # self.data6['x'] = np.array(self.array_nonaligned)
        # self.data_collection.append(self.data6)
        # self.data_collection.add_link(LinkSame(self.data1.world_component_ids[0],
        #                                        self.data6.world_component_ids[1]))
        # self.data_collection.add_link(LinkSame(self.data1.world_component_ids[1],
        #                                        self.data6.world_component_ids[2]))
        # self.data_collection.add_link(LinkSame(self.data1.world_component_ids[2],
        #                                        self.data6.world_component_ids[0]))

    # Start off with the cases where the data is the target data. Enumerate
    # the different cases for the bounds and the expected result.

    DATA_IS_TARGET_CASES = [

        # Bounds are full extent of data
        ([(0, 5, 6), (0, 6, 7), (0, 7, 8), (0, 8, 9)],
            ARRAY),

        # Bounds are inside data
        ([(2, 3, 2), (3, 3, 1), (0, 7, 8), (0, 7, 8)],
            ARRAY[2:4, 3:4, :, :8]),

        # Bounds are outside data along some dimensions
        ([(-5, 9, 15), (3, 5, 3), (0, 9, 10), (5, 6, 2)],
            np.pad(ARRAY[:, 3:6, :, 5:7], [(5, 4), (0, 0), (0, 2), (0, 0)],
                   mode='constant', constant_values=-np.inf)),

        # No overlap
        ([(2, 3, 2), (3, 3, 1), (-5, -4, 2), (0, 7, 8)],
            -np.inf * np.ones((2, 1, 2, 8)))

    ]

    @pytest.mark.parametrize(('bounds', 'expected'), DATA_IS_TARGET_CASES)
    def test_data_is_target_full_bounds(self, bounds, expected):

        buffer = self.data1.compute_fixed_resolution_buffer(target_data=self.data1, bounds=bounds,
                                                            target_cid=self.data1.id['x'])
        assert_equal(buffer, expected)

        buffer = self.data3.compute_fixed_resolution_buffer(target_data=self.data1, bounds=bounds,
                                                            target_cid=self.data3.id['x'])
        assert_equal(buffer, expected)
