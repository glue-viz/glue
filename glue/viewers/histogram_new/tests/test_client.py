# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import pytest
import numpy as np
from mock import MagicMock

from glue.core.subset import RangeSubsetState, CategoricalROISubsetState
from glue.core.component_id import ComponentID
from glue.core.component import CategoricalComponent
from glue.core.data import Data
from glue.core.exceptions import IncompatibleDataException
from glue.core.data_collection import DataCollection
from glue.core.roi import PolygonalROI
from glue.utils import renderless_figure

from ..client import HistogramClient
from ..layer_artist import HistogramLayerArtist


FIGURE = renderless_figure()


class TestHistogramClient(object):

    def setup_method(self, method):
        self.data = Data(x=[0, 0, 0, 1, 2, 3, 3, 10, 20],
                         y=[-1, -1, -1, -2, -2, -2, -3, -5, -7])
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
            if a.y.size > 0:
                datara[0] = min(datara[0], a.y.min())
                datara[1] = max(datara[1], a.y.max())

        assert yra[0] <= datara[0]
        assert yra[1] >= datara[1]

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
        with pytest.raises(IncompatibleDataException):
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
        self.data.new_subset()
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
        comp = self.data.id['x']
        comp2 = self.data.id['y']
        self.client.set_component(comp)
        ct0 = self.draw_count()
        self.client.set_component(comp2)
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
        self.client.autoscale = False
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
        self.client.set_component(self.data.id['y'])
        # bins are -7...-1

        self.data.edit_subset = [self.data.subsets[0]]
        roi = PolygonalROI(vx=[-5.1, -4.5, -3.2], vy=[2, 3, 4])

        self.client.apply_roi(roi)
        state = self.data.subsets[0].subset_state
        assert isinstance(state, RangeSubsetState)

        # range should expand to nearest bin edge
        assert state.lo == -6
        assert state.hi == -3

    def test_apply_roi_xlog(self):
        self.client.add_layer(self.data)
        self.client.set_component(self.data.id['x'])
        self.data.edit_subset = [self.data.subsets[0]]
        self.client.xlog = True
        roi = PolygonalROI(vx=[1, 2, 3], vy=[2, 3, 4])

        self.client.apply_roi(roi)
        state = self.data.subsets[0].subset_state
        assert isinstance(state, RangeSubsetState)
        np.testing.assert_allclose(state.lo, 7.3680629972807736)
        np.testing.assert_allclose(state.hi, 1000)

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
        assert self.client.xlimits == (-7, -1)

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

    def test_artist_clear_resets_arrays(self):
        self.client.add_layer(self.data)
        self.client.set_component(self.data.components[0])
        for a in self.client._artists[self.data]:
            assert a.get_data()[0].size > 0
            a.clear()
            assert a.get_data()[0].size == 0

    def test_component_replaced(self):
        # regression test for 508
        self.client.register_to_hub(self.collect.hub)
        self.client.add_layer(self.data)
        self.client.component = self.data.components[0]

        test = ComponentID('test')
        self.data.update_id(self.client.component, test)
        assert self.client.component is test

    def test_update_when_limits_unchanged(self):

        # Regression test for glue-viz/glue#1010 - this bug caused histograms
        # to not be recomputed if the attribute changed but the limits and
        # number of bins did not.

        self.client.add_layer(self.data)

        self.client.set_component(self.data.id['y'])
        self.client.xlimits = -20, 20
        self.client.nbins = 12

        y1 = self.client._artists[0]._y

        self.client.set_component(self.data.id['x'])
        self.client.xlimits = -20, 20
        self.client.nbins = 12

        y2 = self.client._artists[0]._y
        assert not np.allclose(y1, y2)

        self.client.set_component(self.data.id['y'])

        y3 = self.client._artists[0]._y
        np.testing.assert_allclose(y1, y3)


class TestCategoricalHistogram(TestHistogramClient):

    def setup_method(self, method):
        self.data = Data(y=[-1, -1, -1, -2, -2, -2, -3, -5, -7])
        self.data.add_component(CategoricalComponent(['a', 'a', 'a', 'b', 'c', 'd', 'd', 'e', 'f']), 'x')
        self.subset = self.data.new_subset()
        self.collect = DataCollection(self.data)
        self.client = HistogramClient(self.collect, FIGURE)
        self.axes = self.client.axes
        FIGURE.canvas.draw = MagicMock()
        assert FIGURE.canvas.draw.call_count == 0

    def test_xlimit_single_set(self):
        self.client.add_layer(self.data)
        self.client.set_component(self.data.id['x'])

        self.client.xlimits = (None, 5)
        assert self.client.xlimits == (-0.5, 5)
        self.client.xlimits = (3, None)
        assert self.client.xlimits == (3, 5)

    def test_default_xlimits(self):
        self.client.add_layer(self.data)
        self.client.set_component(self.data.id['x'])
        assert self.client.xlimits == (-0.5, 5.5)
        self.client.set_component(self.data.id['y'])
        assert self.client.xlimits == (-7, -1)

    def test_change_default_bins(self):
        self.client.add_layer(self.data)
        self.client.set_component(self.data.id['x'])
        assert self.client.nbins == 6

    def test_tick_labels(self):
        self.client.add_layer(self.data)
        self.client.set_component(self.data.id['x'])
        correct_labels = ['a', 'b', 'c', 'd', 'e', 'f']
        formatter = self.client.axes.xaxis.get_major_formatter()
        xlabels = [formatter.format_data(pos) for pos in range(6)]
        assert correct_labels == xlabels

    def test_apply_roi(self):
        self.client.add_layer(self.data)
        self.client.set_component(self.data.id['x'])
        # bins are 1...4

        self.data.edit_subset = [self.data.subsets[0]]

        roi = MagicMock()
        roi.to_polygon.return_value = [1.2, 2, 4], [2, 3, 4]

        self.client.apply_roi(roi)
        state = self.data.subsets[0].subset_state
        assert isinstance(state, CategoricalROISubsetState)
        np.testing.assert_equal(self.data.subsets[0].subset_state.roi.categories,
                                np.array(['b', 'c', 'd', 'e']))

    # REMOVED TESTS
    def test_xlog_axes_labels(self):
        """ log-scale doesn't make sense for categorical data"""
        pass

    def test_xlog_snaps_limits(self):
        """ log-scale doesn't make sense for categorical data"""
        pass

    def test_apply_roi_xlog(self):
        """ log-scale doesn't make sense for categorical data"""
        pass

    def test_nbin_override_persists_over_attribute_change(self):
        # regression test for #398
        self.collect.append(self.data)
        self.client.add_layer(self.data)
        self.client.set_component(self.data.id['x'])
        self.client.nbins = 7
        self.client.set_component(self.data.id['y'])
        assert self.client.nbins == 7


class TestCommunication(object):

    def setup_method(self, method):
        self.data = Data(x=[1, 2, 3, 2, 2, 3, 1])
        figure = MagicMock()
        self.collect = DataCollection()
        self.client = HistogramClient(self.collect, figure)
        self.axes = self.client.axes
        self.hub = self.collect.hub
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


class TestHistogramLayerArtist(object):

    def setup_subset(self):
        ax = MagicMock()
        d = Data(x=[1, 2, 3])
        s = d.new_subset()
        s.subset_state = d.id['x'] > 1
        self.artist = HistogramLayerArtist(s, ax)

    def setup_hist_calc_counter(self):
        self.setup_subset()
        m = MagicMock()
        self.artist._calculate_histogram = m
        return m

    def setup_hist_scale_counter(self):
        self.setup_subset()
        m = MagicMock()
        self.artist._scale_histogram = m
        self.artist._calculate_histogram = MagicMock()
        return m

    def test_calculate_histogram_efficient(self):
        ct = self.setup_hist_calc_counter()
        self.artist.update()
        assert ct.call_count == 1
        self.artist.update()
        assert ct.call_count == 1

    def test_recalc_on_state_changes(self):
        ct = self.setup_hist_calc_counter()
        assert ct.call_count == 0
        self.artist.update()
        assert ct.call_count == 1

        # lo
        self.artist.lo -= 1
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 2

        # hi
        self.artist.hi -= 1
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 3

        # nbins
        self.artist.nbins += 1
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 4

        # xlog
        self.artist.xlog ^= True
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 5

        # ylog -- no call
        self.artist.ylog ^= True
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 5

        # cumulative -- no call
        self.artist.cumulative ^= True
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 5

        # normed -- no call
        self.artist.normed ^= True
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 5

        # subset style -- no call
        self.artist.layer.style.color = '#00ff00'
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 5

        # subset state
        self.artist.layer.subset_state = self.artist.layer.data.id['x'] > 10
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 6

    def test_rescale_on_state_changes(self):
        ct = self.setup_hist_scale_counter()
        assert ct.call_count == 0
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 1

        # lo
        self.artist.lo -= 1
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 2

        # hi
        self.artist.hi -= 1
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 3

        # nbins
        self.artist.nbins += 1
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 4

        # xlog
        self.artist.xlog ^= True
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 5

        # ylog
        self.artist.ylog ^= True
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 6

        # cumulative
        self.artist.cumulative ^= True
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 7

        # normed
        self.artist.normed ^= True
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 8

        # subset state
        self.artist.layer.subset_state = self.artist.layer.data.id['x'] > 10
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 9

        # subset style -- no call
        self.artist.layer.style.color = '#00ff00'
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 9
