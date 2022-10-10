import numpy as np
from numpy.testing import assert_allclose

from glue.core.component_id import ComponentID

from ..data_collection import DataCollection
from ..link_helpers import ComponentLink
from ..data import BaseCartesianData
from ..coordinates import IdentityCoordinates


class ExampleBaseData(BaseCartesianData):

    def __init__(self, cid_label='x', coords=None):
        super().__init__(coords=coords)
        self.data_cid = ComponentID(label=cid_label, parent=self)
        self._array = np.arange(12).reshape((1, 4, 3))

    @property
    def label(self):
        return "Example Data"

    @property
    def shape(self):
        return (1, 4, 3)

    @property
    def main_components(self):
        return [self.data_cid]

    def get_kind(self, cid):
        return 'numerical'

    def get_data(self, cid, view=None):
        if cid is self.data_cid:
            if view is None:
                return self._array
            else:
                return self._array[view]
        else:
            return super(ExampleBaseData, self).get_data(cid, view=view)

    def get_mask(self, subset_state, view=None):
        return subset_state.to_mask(self, view=view)

    def compute_statistic(self, statistic, cid,
                          axis=None, finite=True,
                          positive=False, subset_state=None,
                          percentile=None, random_subset=None):
        raise NotImplementedError()

    def compute_histogram(self, cid,
                          range=None, bins=None, log=False,
                          subset_state=None, subset_group=None):
        raise NotImplementedError()


def test_data_coords():

    # Make sure that world_component_ids works in both the case where
    # coords is not defined and when it is defined.

    data1 = ExampleBaseData()
    assert len(data1.pixel_component_ids) == 3
    assert len(data1.world_component_ids) == 0

    data2 = ExampleBaseData(coords=IdentityCoordinates(n_dim=3))
    assert len(data2.pixel_component_ids) == 3
    assert len(data2.world_component_ids) == 3

    for idx in range(3):

        assert_allclose(data2[data2.world_component_ids[idx]],
                        data2[data2.pixel_component_ids[idx]])

        assert_allclose(data2[data2.world_component_ids[idx], (0, slice(2), 2)],
                        data2[data2.pixel_component_ids[idx], (0, slice(2), 2)])


def test_linking():

    data1 = ExampleBaseData(cid_label='x')
    data2 = ExampleBaseData(cid_label='y')
    data3 = ExampleBaseData(cid_label='z')

    dc = DataCollection([data1, data2, data3])

    cid1 = data1.main_components[0]
    cid2 = data2.main_components[0]
    cid3 = data3.main_components[0]

    def double(x):
        return 2 * x

    def halve(x):
        return x / 2.

    link1 = ComponentLink([cid1], cid2, using=double, inverse=halve)
    link2 = ComponentLink([cid2], cid3, using=double, inverse=halve)

    dc.add_link(link1)
    dc.add_link(link2)

    assert_allclose(data1[cid2], data1[cid1] * 2)
    assert_allclose(data1[cid3], data1[cid1] * 4)

    assert_allclose(data2[cid1], data1[cid1] / 2)
    assert_allclose(data3[cid1], data1[cid1] / 4)

    assert_allclose(data2[cid2], data1[cid2] / 2)
    assert_allclose(data3[cid2], data1[cid2] / 4)

    assert_allclose(data2[cid3], data1[cid3] / 2)
    assert_allclose(data3[cid3], data1[cid3] / 4)


def test_pixel_aligned():

    data1 = ExampleBaseData(cid_label='x')
    data2 = ExampleBaseData(cid_label='y')
    data3 = ExampleBaseData(cid_label='z')

    dc = DataCollection([data1, data2, data3])

    def double(x):
        return 2 * x

    def halve(x):
        return x / 2.

    for dim in range(3):

        cid1 = data1.pixel_component_ids[dim]
        cid2 = data2.pixel_component_ids[dim]
        cid3 = data3.pixel_component_ids[dim]

        link1 = ComponentLink([cid1], cid2)

        if dim == 1:
            link2 = ComponentLink([cid2], cid3, using=double, inverse=halve)
        else:
            link2 = ComponentLink([cid2], cid3)

        dc.add_link(link1)
        dc.add_link(link2)

    assert len(data1.pixel_aligned_data) == 1
    assert data1.pixel_aligned_data[data2] == [0, 1, 2]
    assert len(data2.pixel_aligned_data) == 1
    assert data2.pixel_aligned_data[data1] == [0, 1, 2]
    assert len(data3.pixel_aligned_data) == 0
