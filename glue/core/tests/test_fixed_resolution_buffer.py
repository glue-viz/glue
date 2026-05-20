import pytest

import numpy as np
from numpy.testing import assert_equal

from glue.core import Data, DataCollection
from glue.core.link_helpers import LinkSame, ComponentLink
from glue.core.fixed_resolution_buffer import compute_fixed_resolution_buffer

from .test_base_cartesian_data import ExampleBaseData


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
                   mode='constant', constant_values=np.nan)),

        # No overlap
        ([(2, 3, 2), (3, 3, 1), (-5, -4, 2), (0, 7, 8)],
            np.nan * np.ones((2, 1, 2, 8)))

    ]

    @pytest.mark.parametrize(('bounds', 'expected'), DATA_IS_TARGET_CASES)
    def test_data_is_target_full_bounds(self, bounds, expected):

        buffer = self.data1.compute_fixed_resolution_buffer(target_data=self.data1, bounds=bounds,
                                                            target_cid=self.data1.id['x'])
        assert_equal(buffer, expected)

        buffer = self.data3.compute_fixed_resolution_buffer(target_data=self.data1, bounds=bounds,
                                                            target_cid=self.data3.id['x'])
        assert_equal(buffer, expected)


def test_base_cartesian_data():

    # Make sure that things work properly when using a BaseCartesianData
    # subclass that isn't necessarily Data.

    data1 = ExampleBaseData(cid_label='x')
    data2 = ExampleBaseData(cid_label='y')

    dc = DataCollection([data1, data2])

    def add_one(x):
        return x + 1

    def sub_one(x):
        return x - 1

    link1 = ComponentLink([data1.pixel_component_ids[0]], data2.pixel_component_ids[0])
    link2 = ComponentLink([data1.pixel_component_ids[1]], data2.pixel_component_ids[1], using=add_one, inverse=sub_one)
    link3 = ComponentLink([data1.pixel_component_ids[2]], data2.pixel_component_ids[2], using=sub_one, inverse=add_one)

    dc.add_link(link1)
    dc.add_link(link2)
    dc.add_link(link3)

    assert_equal(compute_fixed_resolution_buffer(data1,
                                                 target_data=data1,
                                                 bounds=[(-1, 1, 3), (0, 3, 4), 1],
                                                 target_cid=data1.main_components[0]),
                 np.array([[np.nan, np.nan, np.nan, np.nan],
                           [1, 4, 7, 10],
                           [np.nan, np.nan, np.nan, np.nan]]))

    assert_equal(compute_fixed_resolution_buffer(data2,
                                                 target_data=data2,
                                                 bounds=[(-1, 1, 3), (0, 3, 4), 1],
                                                 target_cid=data2.main_components[0]),
                 np.array([[np.nan, np.nan, np.nan, np.nan],
                           [1, 4, 7, 10],
                           [np.nan, np.nan, np.nan, np.nan]]))

    assert_equal(compute_fixed_resolution_buffer(data1,
                                                 target_data=data2,
                                                 bounds=[(-1, 1, 3), (0, 3, 4), 1],
                                                 target_cid=data1.main_components[0]),
                 np.array([[np.nan, np.nan, np.nan, np.nan],
                           [np.nan, 2, 5, 8],
                           [np.nan, np.nan, np.nan, np.nan]]))


# ---------------------------------------------------------------------------
# invalidate_cache helper
# ---------------------------------------------------------------------------


class TestInvalidateCache():

    def setup_method(self, method):
        from glue.core import fixed_resolution_buffer as frb_mod
        self.frb_mod = frb_mod
        # Stash the real caches so we can restore them.
        self._array_backup = dict(frb_mod.ARRAY_CACHE)
        self._pixel_backup = dict(frb_mod.PIXEL_CACHE)
        frb_mod.ARRAY_CACHE.clear()
        frb_mod.PIXEL_CACHE.clear()

    def teardown_method(self, method):
        self.frb_mod.ARRAY_CACHE.clear()
        self.frb_mod.PIXEL_CACHE.clear()
        self.frb_mod.ARRAY_CACHE.update(self._array_backup)
        self.frb_mod.PIXEL_CACHE.update(self._pixel_backup)

    def test_invalidate_specific_data(self):
        from glue.core.fixed_resolution_buffer import invalidate_cache
        d1 = object()
        d2 = object()
        self.frb_mod.ARRAY_CACHE['k1'] = {'hash': (d1, None, None, None, True),
                                          'array': None}
        self.frb_mod.ARRAY_CACHE['k2'] = {'hash': (d2, None, None, None, True),
                                          'array': None}
        self.frb_mod.PIXEL_CACHE['k1'] = {'hash': (d1, None)}
        invalidate_cache(d1)
        assert 'k1' not in self.frb_mod.ARRAY_CACHE
        assert 'k1' not in self.frb_mod.PIXEL_CACHE
        assert 'k2' in self.frb_mod.ARRAY_CACHE

    def test_invalidate_target_data_slot(self):
        from glue.core.fixed_resolution_buffer import invalidate_cache
        d = object()
        # `d` appears in the target_data position of the hash.
        self.frb_mod.ARRAY_CACHE['k'] = {'hash': (object(), None, d, None, True),
                                         'array': None}
        invalidate_cache(d)
        assert 'k' not in self.frb_mod.ARRAY_CACHE

    def test_invalidate_all(self):
        from glue.core.fixed_resolution_buffer import invalidate_cache
        self.frb_mod.ARRAY_CACHE['k1'] = {'hash': (object(),), 'array': None}
        self.frb_mod.PIXEL_CACHE['k2'] = {'hash': (object(),)}
        invalidate_cache()
        assert self.frb_mod.ARRAY_CACHE == {}
        assert self.frb_mod.PIXEL_CACHE == {}

    def test_invalidate_tolerates_malformed_entries(self):
        from glue.core.fixed_resolution_buffer import invalidate_cache
        d = object()
        self.frb_mod.ARRAY_CACHE['no-hash'] = {'array': None}
        self.frb_mod.ARRAY_CACHE['ok'] = {'hash': (d,), 'array': None}
        invalidate_cache(d)
        # Malformed entry is left alone; matched entry is dropped.
        assert 'no-hash' in self.frb_mod.ARRAY_CACHE
        assert 'ok' not in self.frb_mod.ARRAY_CACHE
