#pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103
import pytest

from mock import MagicMock
import matplotlib.pyplot as plt
import numpy as np

from ..histogram_client import HistogramClient

from ...core.data_collection import DataCollection
from ...core.exceptions import IncompatibleDataException
from ...core.hub import Hub
from ...core.data import Data
from ...core.subset import RangeSubsetState

FIGURE = plt.figure()
plt.close('all')


class TestException(Exception):
    pass


class TestHistogramClient(object):

    def setup_method(self, method):
        self.data = Data(x=[0, 0, 0, 1, 2, 3, 3, 10, 20],
                         y=[-1, -1, -1, -2, -2, -2, -3, -5, -10])
        self.subset = self.data.new_subset()
        self.collect = DataCollection(self.data)
        self.client = HistogramClient(self.collect, FIGURE)
        self.axes = self.client.axes
        FIGURE.canvas.draw = MagicMock()
        assert FIGURE.canvas.draw.call_count == 0

    def draw_count(self):
        return self.axes.figure.canvas.draw.call_count

    def layer_drawn(self, layer):
        return layer in self.client._artists and \
            all(a.visible for a in self.client._artists[layer]) and \
            all(len(a.artists) > 0 for a in self.client._artists[layer])

    def layer_present(self, layer):
        return layer in self.client._artists

    def assert_autoscaled(self):
        yra = self.client.axes.get_ylim()
        datara = [99999, -99999]
        for a in self.client._artists:
            y = a.y
            if a.y.size > 0:
                datara[0] = min(datara[0], a.y.min())
                datara[1] = max(datara[1], a.y.max())

        np.testing.assert_array_almost_equal(yra[0], 0)
        np.testing.assert_array_almost_equal(datara[1], yra[1])

    def test_empty_on_creation(self):
        assert self.data not in self.client._artists

    def test_add_layer(self):
        self.client.add_layer(self.data)
        assert self.layer_present(self.data)
        assert not self.layer_drawn(self.data)

        self.client.set_component(self.data.components[0])
        assert self.layer_drawn(self.data)

    def test_add_invalid_layer_raises(self):
        self.collect.remove(self.data)
        with pytest.raises(IncompatibleDataException) as exc:
            self.client.add_layer(self.data)

    def test_add_subset_auto_adds_data(self):
        subset = self.data.new_subset()
        self.client.add_layer(subset)
        assert self.layer_present(self.data)
        assert self.layer_present(subset)

        self.client.set_component(self.data.components[0])
        assert self.layer_drawn(self.data)

    def test_double_add_ignored(self):
        self.client.add_layer(self.data)
        art = self.client._artists[self.data]
        self.client.add_layer(self.data)
        assert self.client._artists[self.data] == art

    def test_add_data_auto_adds_subsets(self):
        s = self.data.new_subset()
        self.client.add_layer(self.data)
        assert self.layer_present(s)

    def test_data_removal(self):
        self.client.add_layer(self.data)
        self.client.remove_layer(self.data)
        assert not (self.layer_present(self.data))

    def test_data_removal_removes_subsets(self):
        self.client.add_layer(self.data)
        self.client.remove_layer(self.data)
        s = self.data.new_subset()
        assert len(self.data.subsets) > 0

        for subset in self.data.subsets:
            assert not (self.layer_present(subset))

    def test_layer_updates_on_data_add(self):
        self.client.add_layer(self.data)
        for s in self.data.subsets:
            assert s in self.client._artists

    def test_set_component_updates_component(self):
        self.client.add_layer(self.data)
        comp = self.data.find_component_id('uniform')
        self.client.set_component(comp)
        assert self.client._component is comp

    def test_set_component_redraws(self):
        self.client.add_layer(self.data)
        comp = self.data.find_component_id('uniform')
        ct0 = self.draw_count()
        self.client.set_component(comp)
        assert self.draw_count() > ct0

    def test_remove_not_present_ignored(self):
        self.client.remove_layer(self.data)

    def test_set_visible_external_data(self):
        self.client.set_layer_visible(None, False)

    def test_get_visible_external_data(self):
        assert not (self.client.is_layer_visible(None))

    def test_set_visible(self):
        self.client.add_layer(self.data)
        self.client.set_layer_visible(self.data, False)
        assert not (self.client.is_layer_visible(self.data))

    def test_draw_histogram_one_layer(self):
        self.client.add_layer(self.data)
        self.client.set_component(self.data.find_component_id('uniform'))

    def test_draw_histogram_subset_hidden(self):
        self.client.add_layer(self.data)
        s = self.data.new_subset()
        self.client.set_layer_visible(s, False)
        self.client.set_component(self.data.find_component_id('uniform'))

    def test_draw_histogram_two_layers(self):
        self.client.add_layer(self.data)
        self.client.set_component(self.data.find_component_id('uniform'))

    def test_update_property_set_triggers_redraw(self):
        self.client.add_layer(self.data)
        ct = self.draw_count()
        self.client.normed ^= True
        assert self.draw_count() > ct

    @pytest.mark.parametrize(('prop'), ['normed', 'cumulative'])
    def test_set_boolean_property(self, prop):
        """Boolean properties should sync with artists"""
        self.client.add_layer(self.data)
        self.client.set_component(self.data.components[0])

        setattr(self.client, prop, False)
        for a in self.client._artists:
            assert not getattr(a, prop)

        setattr(self.client, prop, True)
        for a in self.client._artists:
            assert getattr(a, prop)

    def test_set_nbins(self):
        self.client.add_layer(self.data)
        self.client.set_component(self.data.components[0])

        self.client.nbins = 100
        for a in self.client._artists[self.data]:
            assert a.nbins == 100
            assert a.x.size == 100 + 1

    def test_autoscale(self):
        self.client.add_layer(self.data)
        self.client.set_component(self.data.components[0])
        self.client.axes.set_ylim(0, .1)
        self.client.autoscale = True
        self.assert_autoscaled()

    def test_xlimits(self):
        self.client.add_layer(self.data)
        self.client.set_component(self.data.components[0])

        self.client.xlimits = -12, 20
        assert self.client.xlimits == (-12, 20)
        for a in self.client._artists[self.data]:
            assert a.lo == -12
            assert a.hi == 20

    def test_set_xlimits_out_of_data_range(self):
        """Setting xlimits outside of range shouldn't crash"""
        self.client.add_layer(self.data)
        self.client.set_component(self.data.components[0])

        self.client.xlimits = 100, 200
        self.client.xlimits = -200, -100

    def test_component_property(self):
        self.client.add_layer(self.data)
        self.client.set_component(self.data.components[0])
        assert self.client.component is self.data.components[0]

    def test_apply_roi(self):
        self.client.add_layer(self.data)
        self.data.edit_subset = [self.data.subsets[0]]
        roi = MagicMock()
        roi.to_polygon.return_value = [1, 2, 3], [2, 3, 4]

        self.client.apply_roi(roi)
        state = self.data.subsets[0].subset_state
        assert isinstance(state, RangeSubsetState)
        assert state.lo == 1
        assert state.hi == 3

    def test_apply_roi_xlog(self):
        self.client.add_layer(self.data)
        self.data.edit_subset = [self.data.subsets[0]]
        self.client.xlog = True
        roi = MagicMock()
        roi.to_polygon.return_value = [1, 2, 3], [2, 3, 4]

        self.client.apply_roi(roi)
        state = self.data.subsets[0].subset_state
        assert isinstance(state, RangeSubsetState)
        assert state.lo == 10
        assert state.hi == 1000

    def test_xlimits_sticky_with_component(self):
        self.client.add_layer(self.data)
        self.client.set_component(self.data.components[0])
        self.client.xlimits = 5, 6

        self.client.set_component(self.data.components[1])
        self.client.xlimits = 7, 8

        self.client.set_component(self.data.components[0])
        assert self.client.xlimits == (5, 6)

        self.client.set_component(self.data.components[1])
        assert self.client.xlimits == (7, 8)

    def test_default_xlimits(self):
        self.client.add_layer(self.data)
        self.client.set_component(self.data.id['x'])
        assert self.client.xlimits == (0, 20)
        self.client.set_component(self.data.id['y'])
        assert self.client.xlimits == (-10, -1)

    def test_xlimit_single_set(self):
        self.client.add_layer(self.data)
        self.client.set_component(self.data.id['x'])

        self.client.xlimits = (None, 5)
        assert self.client.xlimits == (0, 5)
        self.client.xlimits = (3, None)
        assert self.client.xlimits == (3, 5)

    def test_xlimit_reverse_set(self):
        self.client.add_layer(self.data)
        self.client.set_component(self.data.id['x'])

        self.client.xlimits = 5, 3
        assert self.client.xlimits == (3, 5)

    def test_xlog_axes_labels(self):
        self.client.add_layer(self.data)
        self.client.set_component(self.data.id['x'])

        self.client.xlog = True
        assert self.client.axes.get_xlabel() == 'Log x'

        self.client.xlog = False
        assert self.client.axes.get_xlabel() == 'x'

        self.client.ylog = True
        assert self.client.axes.get_ylabel() == 'N'

        self.client.ylog = False
        assert self.client.axes.get_ylabel() == 'N'

    def test_xlog_snaps_limits(self):

        self.client.add_layer(self.data)
        self.client.set_component(self.data.id['x'])

        self.client.axes.set_xlim((-1, 1))
        self.client.xlog = True
        assert self.client.axes.get_xlim() != (-1, 1)


class TestCommunication(object):
    def setup_method(self, method):
        self.data = Data(x=[1, 2, 3, 2, 2, 3, 1])
        figure = MagicMock()
        self.collect = DataCollection()
        self.client = HistogramClient(self.collect, figure)
        self.axes = self.client.axes
        self.hub = Hub()
        self.connect()

    def draw_count(self):
        return self.axes.figure.canvas.draw.call_count

    def connect(self):
        self.client.register_to_hub(self.hub)
        self.collect.register_to_hub(self.hub)

    def test_ignore_data_add_message(self):
        self.collect.append(self.data)
        assert not (self.client.layer_present(self.data))

    def test_update_data_ignored_if_data_not_present(self):
        self.collect.append(self.data)
        ct0 = self.draw_count()
        self.data.style.color = 'blue'
        assert self.draw_count() == ct0

    def test_update_data_processed_if_data_present(self):
        self.collect.append(self.data)
        self.client.add_layer(self.data)
        ct0 = self.draw_count()
        self.data.style.color = 'blue'
        assert self.draw_count() > ct0

    def test_add_subset_ignored_if_data_not_present(self):
        self.collect.append(self.data)
        ct0 = self.draw_count()
        sub = self.data.new_subset()
        assert not (self.client.layer_present(sub))

    def test_add_subset_processed_if_data_present(self):
        self.collect.append(self.data)
        self.client.add_layer(self.data)
        sub = self.data.new_subset()
        assert (self.client.layer_present(sub))

    def test_update_subset_ignored_if_not_present(self):
        self.collect.append(self.data)
        self.client.add_layer(self.data)
        sub = self.data.new_subset()
        self.client.remove_layer(sub)
        ct0 = self.draw_count()
        sub.style.color = 'blue'
        assert self.draw_count() == ct0

    def test_update_subset_processed_if_present(self):
        self.collect.append(self.data)
        self.client.add_layer(self.data)
        sub = self.data.new_subset()
        ct0 = self.draw_count()
        sub.style.color = 'blue'
        assert self.draw_count() > ct0

    def test_data_remove_message(self):
        self.collect.append(self.data)
        self.client.add_layer(self.data)
        self.collect.remove(self.data)
        assert not self.client.layer_present(self.data)

    def test_subset_remove_message(self):
        self.collect.append(self.data)
        self.client.add_layer(self.data)
        sub = self.data.new_subset()
        assert self.client.layer_present(sub)
        sub.delete()
        assert not self.client.layer_present(sub)
