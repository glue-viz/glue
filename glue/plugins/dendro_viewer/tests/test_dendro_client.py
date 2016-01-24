from __future__ import absolute_import, division, print_function

from mock import MagicMock
from numpy.testing import assert_array_equal

from glue.core.edit_subset_mode import EditSubsetMode
from glue.core.roi import PointROI
from glue.core import Data, DataCollection
from glue.utils import renderless_figure

from ..client import DendroClient


# share matplotlib instance, and disable rendering, for speed
FIGURE = renderless_figure()


class TestDendroClient():

    def setup_method(self, method):

        self.data = Data(parent=[4, 4, 5, 5, 5, -1],
                         height=[5, 4, 3, 2, 1, 0],
                         label='dendro')
        self.dc = DataCollection([self.data])
        self.hub = self.dc.hub
        self.client = DendroClient(self.dc, figure=FIGURE)
        EditSubsetMode().data_collection = self.dc

    def add_subset_via_hub(self):
        self.connect()
        self.client.add_layer(self.data)
        s = self.data.new_subset()
        return s

    def connect(self):
        self.client.register_to_hub(self.hub)
        self.dc.register_to_hub(self.hub)

    def click(self, x, y):
        roi = PointROI(x=x, y=y)
        self.client.apply_roi(roi)

    def test_data_present_after_adding(self):
        assert self.data not in self.client
        self.client.add_layer(self.data)
        assert self.data in self.client

    def test_add_data_adds_subsets(self):
        s1 = self.data.new_subset()
        self.client.add_layer(self.data)
        assert s1 in self.client

    def test_remove_data(self):

        self.client.add_layer(self.data)
        self.client.remove_layer(self.data)

        assert self.data not in self.client

    def test_remove_data_removes_subsets(self):
        s = self.data.new_subset()
        self.client.add_layer(self.data)

        self.client.remove_layer(self.data)
        assert s not in self.client

    def test_add_subset_hub(self):
        s = self.add_subset_via_hub()
        assert s in self.client

    def test_new_subset_autoadd(self):
        self.connect()
        self.client.add_layer(self.data)
        s = self.data.new_subset()
        assert s in self.client

    def test_remove_subset_hub(self):

        s = self.add_subset_via_hub()
        s.delete()

        assert s not in self.client

    def test_subset_sync(self):
        s = self.add_subset_via_hub()

        self.client._update_layer = MagicMock()
        s.style.color = 'blue'
        self.client._update_layer.assert_called_once_with(s)

    def test_data_sync(self):
        self.connect()
        self.client.add_layer(self.data)

        self.client._update_layer = MagicMock()
        self.data.style.color = 'blue'
        self.client._update_layer.assert_called_once_with(self.data)

    def test_data_remove(self):
        s = self.add_subset_via_hub()
        self.dc.remove(self.data)

        assert self.data not in self.dc
        assert self.data not in self.client
        assert s not in self.client

    def test_log(self):
        self.client.ylog = True
        assert self.client.axes.get_yscale() == 'log'

    def test_1d_data_required(self):
        d = Data(x=[[1, 2], [2, 3]])
        self.dc.append(d)
        self.client.add_layer(d)
        assert d not in self.client

    def test_apply_roi(self):
        self.client.add_layer(self.data)
        self.client.select_substruct = False

        self.click(0, 4)
        s = self.data.subsets[0]

        assert_array_equal(s.to_index_list(), [1])

        self.click(0, 3)
        assert_array_equal(s.to_index_list(), [1])

        self.click(0, 0)
        assert_array_equal(s.to_index_list(), [4])

        self.click(.75, 4)
        assert_array_equal(s.to_index_list(), [0])

        self.click(0, 10)
        assert_array_equal(s.to_index_list(), [])

    def test_apply_roi_children_select(self):
        self.client.select_substruct = True
        self.client.add_layer(self.data)

        self.click(.5, .5)
        s = self.data.subsets[0]

        assert_array_equal(s.to_index_list(), [0, 1, 4])

    def test_attribute_change_triggers_relayout(self):
        self.client.add_layer(self.data)

        l = self.client._layout
        self.client.height_attr = self.data.id['parent']
        assert self.client._layout is not l

        l = self.client._layout
        self.client.parent_attr = self.data.id['height']
        assert self.client._layout is not l

        l = self.client._layout
        self.client.order_attr = self.data.id['parent']
        assert self.client._layout is not l
