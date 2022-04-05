# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

import pytest

import numpy as np
from numpy.testing import assert_array_equal

from glue.core.data import Data
from glue.core.data_collection import DataCollection
from glue.core.link_helpers import LinkSame, MultiLink
from glue.core.exceptions import IncompatibleAttribute

from ..pixel_selection_subset_state import PixelSubsetState


def test_pixel_selection_subset_state():

    data1 = Data(x=np.ones((2, 4, 3)))
    data2 = Data(y=np.ones((4, 3, 2)))
    data3 = Data(z=np.ones((2, 3)))

    y_id = data2.main_components[0]
    z_id = data3.main_components[0]

    slice_1d = [slice(1, 2), slice(None), slice(None)]
    slice_2d = [slice(None), slice(2, 3), slice(1, 2)]
    slice_3d = [slice(1, 2), slice(2, 3), slice(1, 2)]

    state_1d = PixelSubsetState(data1, slice_1d)
    state_2d = PixelSubsetState(data1, slice_2d)
    state_3d = PixelSubsetState(data1, slice_3d)

    states = [state_1d, state_2d, state_3d]

    dc = DataCollection([data1, data2, data3])

    # Calling to_array with reference data should work by default, and not work
    # with unlinked datasets.

    for data in dc:
        for state in states:
            cid = data.main_components[0]
            if data is data1:
                assert_array_equal(state.to_array(data, cid), data[cid][tuple(state.slices)])
            else:
                with pytest.raises(IncompatibleAttribute):
                    state.to_array(data, cid)

    # Add one set of links

    dc.add_link(LinkSame(data1.pixel_component_ids[0], data2.pixel_component_ids[2]))
    dc.add_link(LinkSame(data1.pixel_component_ids[0], data3.pixel_component_ids[0]))

    assert_array_equal(state_1d.to_array(data2, y_id), data2[y_id][:, :, 1:2])
    with pytest.raises(IncompatibleAttribute):
        state_2d.to_array(data2, y_id)
    with pytest.raises(IncompatibleAttribute):
        state_3d.to_array(data2, y_id)

    assert_array_equal(state_1d.to_array(data3, z_id), data3[z_id][1:2])
    with pytest.raises(IncompatibleAttribute):
        state_2d.to_array(data3, z_id)
    with pytest.raises(IncompatibleAttribute):
        state_3d.to_array(data3, z_id)

    # Add links with multiple components, in this case linking two cids with two cids

    def forwards(x, y):
        return x + y, x - y

    def backwards(x, y):
        return 0.5 * (x + y), 0.5 * (x - y)

    dc.add_link(MultiLink(data1.pixel_component_ids[1:],
                          data2.pixel_component_ids[:2],
                          forwards=forwards, backwards=backwards))

    assert_array_equal(state_1d.to_array(data2, y_id), data2[y_id][:, :, 1:2])
    assert_array_equal(state_2d.to_array(data2, y_id), data2[y_id][2:3, 1:2, :])
    assert_array_equal(state_3d.to_array(data2, y_id), data2[y_id][2:3, 1:2, 1:2])

    assert_array_equal(state_1d.to_array(data3, z_id), data3[z_id][1:2])
    with pytest.raises(IncompatibleAttribute):
        state_2d.to_array(data3, z_id)
    with pytest.raises(IncompatibleAttribute):
        state_3d.to_array(data3, z_id)
