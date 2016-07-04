# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

from timeit import timeit
from functools import partial

import pytest
import numpy as np
from mock import MagicMock
from matplotlib.ticker import AutoLocator, MaxNLocator, LogLocator
from matplotlib.ticker import LogFormatterMathtext, ScalarFormatter, FuncFormatter

from glue.core.edit_subset_mode import EditSubsetMode
from glue.core.component_id import ComponentID
from glue.core.component import Component, CategoricalComponent
from glue.core.data_collection import DataCollection
from glue.core.data import Data
from glue.core.roi import RectangularROI, XRangeROI, YRangeROI
from glue.core.subset import (RangeSubsetState, CategoricalROISubsetState,
                              AndState)
from glue.tests import example_data
from glue.utils import renderless_figure

from ..client import ScatterClient


# share matplotlib instance, and disable rendering, for speed
FIGURE = renderless_figure()


class TestScatterClient(object):

    def setup_method(self, method):
        self.data = example_data.test_data()
        self.ids = [self.data[0].find_component_id('a'),
                    self.data[0].find_component_id('b'),
                    self.data[1].find_component_id('c'),
                    self.data[1].find_component_id('d')]
        self.roi_limits = (0.5, 0.5, 1.5, 1.5)
        self.roi_points = (np.array([1]), np.array([1]))
        self.collect = DataCollection()
        EditSubsetMode().data_collection = self.collect

        self.hub = self.collect.hub

        FIGURE.clf()
        axes = FIGURE.add_subplot(111)
        self.client = ScatterClient(self.collect, axes=axes)

        self.connect()

    def teardown_method(self, methdod):
        self.assert_properties_correct()
        self.assert_axes_ticks_correct()

    def assert_properties_correct(self):
        ax = self.client.axes
        cl = self.client
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert abs(cl.xmin - min(xlim)) < 1e-2
        assert abs(cl.xmax - max(xlim)) < 1e-2
        assert abs(cl.ymin - min(ylim)) < 1e-2
        assert abs(cl.ymax - max(ylim)) < 1e-2
        assert cl.xflip == (xlim[1] < xlim[0])
        assert cl.yflip == (ylim[1] < ylim[0])
        assert cl.xlog == (ax.get_xscale() == 'log')
        assert cl.ylog == (ax.get_yscale() == 'log')
        assert (self.client.xatt is None) or isinstance(
            self.client.xatt, ComponentID)
        assert (self.client.yatt is None) or isinstance(
            self.client.yatt, ComponentID)

    def check_ticks(self, axis, is_log, is_cat):
        locator = axis.get_major_locator()
        formatter = axis.get_major_formatter()
        if is_log:
            assert isinstance(locator, LogLocator)
            assert isinstance(formatter, LogFormatterMathtext)
        elif is_cat:
            assert isinstance(locator, MaxNLocator)
            assert isinstance(formatter, FuncFormatter)
        else:
            assert isinstance(locator, AutoLocator)
            assert isinstance(formatter, ScalarFormatter)

    def assert_axes_ticks_correct(self):
        ax = self.client.axes
        client = self.client
        if client.xatt is not None:
            self.check_ticks(ax.xaxis,
                             client.xlog,
                             client._check_categorical(client.xatt))
        if client.yatt is not None:
            self.check_ticks(ax.yaxis,
                             client.ylog,
                             client._check_categorical(client.yatt))

    def plot_data(self, layer):
        """ Return the data bounds for a given layer (data or subset)
        Output format: [xmin, xmax], [ymin, ymax]
        """
        client = self.client
        x, y = client.artists[layer][0].get_data()
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()
        return [xmin, xmax], [ymin, ymax]

    def plot_limits(self):
        """ Return the plot limits
        Output format [xmin, xmax], [ymin, ymax]
        """
        ax = self.client.axes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        return (min(xlim), max(xlim)), (min(ylim), max(ylim))

    def assert_layer_inside_limits(self, layer):
        """Assert that points of a layer are within plot limits """
        xydata = self.plot_data(layer)
        xylimits = self.plot_limits()
        assert xydata[0][0] >= xylimits[0][0]
        assert xydata[1][0] >= xylimits[1][0]
        assert xydata[0][1] <= xylimits[0][1]
        assert xydata[1][1] <= xylimits[1][1]

    def setup_2d_data(self):
        d = Data(x=[[1, 2], [3, 4]], y=[[2, 4], [6, 8]])
        self.collect.append(d)
        self.client.add_layer(d)
        self.client.xatt = d.id['x']
        self.client.yatt = d.id['y']
        return d

    def add_data(self, data=None):
        if data is None:
            data = self.data[0]
        data.edit_subset = data.new_subset()
        self.collect.append(data)
        self.client.add_data(data)
        return data

    def add_data_and_attributes(self):
        data = self.add_data()
        data.edit_subset = data.new_subset()
        self.client.xatt = self.ids[0]
        self.client.yatt = self.ids[1]
        return data

    def is_first_in_front(self, front, back):
        z1 = self.client.get_layer_order(front)
        z2 = self.client.get_layer_order(back)
        return z1 > z2

    def connect(self):
        self.client.register_to_hub(self.hub)
        self.collect.register_to_hub(self.hub)

    def layer_drawn(self, layer):
        return self.client.is_layer_present(layer) and \
            all(a.enabled and a.visible for a in self.client.artists[layer])

    def layer_data_correct(self, layer, x, y):
        xx, yy = self.client.artists[layer][0].get_data()
        if max(abs(xx - x)) > .01:
            return False
        if max(abs(yy - y)) > .01:
            return False
        return True

    def test_empty_on_creation(self):
        for d in self.data:
            assert not self.client.is_layer_present(d)

    def test_add_external_data_raises_exception(self):
        data = Data()
        with pytest.raises(TypeError) as exc:
            self.client.add_data(data)
        assert exc.value.args[0] == "Layer not in data collection"

    def test_valid_add(self):
        self.add_data()
        assert self.client.is_layer_present(self.data[0])

    def test_axis_labels_sync_with_setters(self):
        self.add_data()
        self.client.xatt = self.ids[1]
        assert self.client.axes.get_xlabel() == self.ids[1].label
        self.client.yatt = self.ids[0]
        assert self.client.axes.get_ylabel() == self.ids[0].label

    def test_setters_require_componentID(self):
        self.add_data()
        with pytest.raises(TypeError):
            self.client.xatt = self.ids[1]._label
        self.client.xatt = self.ids[1]

    def test_logs(self):
        self.add_data()
        self.client.xlog = True
        assert self.client.axes.get_xscale() == 'log'

        self.client.xlog = False
        assert self.client.axes.get_xscale() == 'linear'

        self.client.ylog = True
        assert self.client.axes.get_yscale() == 'log'

        self.client.ylog = False
        assert self.client.axes.get_yscale() == 'linear'

    def test_flips(self):
        self.add_data()

        self.client.xflip = True
        self.assert_flips(True, False)

        self.client.xflip = False
        self.assert_flips(False, False)

        self.client.yflip = True
        self.assert_flips(False, True)

        self.client.yflip = False
        self.assert_flips(False, False)

    def assert_flips(self, xflip, yflip):
        ax = self.client.axes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert (xlim[1] < xlim[0]) == xflip
        assert (ylim[1] < ylim[0]) == yflip

    def test_double_add(self):
        n0 = len(self.client.axes.lines)
        layer = self.add_data_and_attributes()
        # data present
        assert len(self.client.axes.lines) == n0 + 1 + len(layer.subsets)
        layer = self.add_data()
        # data still present
        assert len(self.client.axes.lines) == n0 + 1 + len(layer.subsets)

    def test_data_updates_propagate(self):
        layer = self.add_data_and_attributes()
        assert self.layer_drawn(layer)
        self.client._layer_updated = False
        layer.style.color = 'k'
        assert self.client._layer_updated

    def test_data_removal(self):
        layer = self.add_data()
        subset = layer.new_subset()
        self.collect.remove(layer)
        assert not self.client.is_layer_present(layer)
        assert not self.client.is_layer_present(subset)

    def test_add_subset_while_connected(self):
        layer = self.add_data()
        subset = layer.new_subset()
        assert self.client.is_layer_present(subset)

    def test_subset_removal(self):
        layer = self.add_data()
        subset = layer.new_subset()
        assert self.client.is_layer_present(layer)
        subset.delete()
        assert not self.client.is_layer_present(subset)

    def test_subset_removal_removes_from_plot(self):
        layer = self.add_data_and_attributes()
        subset = layer.new_subset()
        ct0 = len(self.client.axes.lines)
        subset.delete()
        assert len(self.client.axes.lines) == ct0 - 1

    def test_add_subset_to_untracked_data(self):
        subset = self.data[0].new_subset()
        assert not self.client.is_layer_present(subset)

    def test_valid_plot_data(self):
        layer = self.add_data_and_attributes()
        x = layer[self.ids[0]]
        y = layer[self.ids[1]]
        assert self.layer_data_correct(layer, x, y)

    def test_attribute_update_plot_data(self):
        layer = self.add_data_and_attributes()
        x = layer[self.ids[0]]
        y = layer[self.ids[0]]
        self.client.yatt = self.ids[0]
        assert self.layer_data_correct(layer, x, y)

    def test_invalid_plot(self):
        layer = self.add_data_and_attributes()
        assert self.layer_drawn(layer)
        c = ComponentID('bad id')
        self.client.xatt = c
        assert not self.layer_drawn(layer)
        self.client.xatt = self.ids[0]

    def test_redraw_called_on_invalid_plot(self):
        """ Plot should be updated when given invalid data,
        to sync layers' disabled/invisible states"""
        ctr = MagicMock()
        layer = self.add_data_and_attributes()
        assert self.layer_drawn(layer)
        c = ComponentID('bad id')
        self.client._redraw = ctr
        ct0 = ctr.call_count
        self.client.xatt = c
        ct1 = ctr.call_count
        ncall = ct1 - ct0
        expected = len(self.client.artists)
        assert ncall >= expected
        self.client.xatt = self.ids[0]

    def test_two_incompatible_data(self):
        d0 = self.add_data(self.data[0])
        d1 = self.add_data(self.data[1])
        self.client.xatt = self.ids[0]
        self.client.yatt = self.ids[1]
        x = d0[self.ids[0]]
        y = d0[self.ids[1]]
        assert self.layer_drawn(d0)
        assert self.layer_data_correct(d0, x, y)
        assert not self.layer_drawn(d1)

        self.client.xatt = self.ids[2]
        self.client.yatt = self.ids[3]
        x = d1[self.ids[2]]
        y = d1[self.ids[3]]
        assert self.layer_drawn(d1)
        assert self.layer_data_correct(d1, x, y)
        assert not self.layer_drawn(d0)

    def test_subsets_connect_with_data(self):
        data = self.data[0]
        s1 = data.new_subset()
        s2 = data.new_subset()
        self.collect.append(data)
        self.client.add_data(data)
        assert self.client.is_layer_present(s1)
        assert self.client.is_layer_present(s2)
        assert self.client.is_layer_present(data)

        # should also work with add_layer
        self.collect.remove(data)
        assert data not in self.collect
        assert not self.client.is_layer_present(s1)
        self.collect.append(data)
        self.client.add_layer(data)
        assert self.client.is_layer_present(s1)

    def test_edit_subset_connect_with_data(self):
        data = self.add_data()
        assert self.client.is_layer_present(data.edit_subset)

    def test_edit_subset_removed_with_data(self):
        data = self.add_data()
        self.collect.remove(data)
        assert not self.client.is_layer_present(data.edit_subset)

    def test_apply_roi(self):
        data = self.add_data_and_attributes()
        roi = RectangularROI()
        roi.update_limits(*self.roi_limits)
        x, y = self.roi_points
        self.client.apply_roi(roi)
        assert self.layer_data_correct(data.edit_subset, x, y)

    def test_apply_roi_adds_on_empty(self):
        data = self.add_data_and_attributes()
        data._subsets = []
        data.edit_subset = None
        roi = RectangularROI()
        roi.update_limits(*self.roi_limits)
        x, y = self.roi_points
        self.client.apply_roi(roi)
        assert data.edit_subset is not None

    def test_apply_roi_applies_to_all_editable_subsets(self):
        d1 = self.add_data_and_attributes()
        d2 = self.add_data()
        state1 = d1.edit_subset.subset_state
        state2 = d2.edit_subset.subset_state
        roi = RectangularROI()
        roi.update_limits(*self.roi_limits)
        x, y = self.roi_points
        self.client.apply_roi(roi)
        assert d1.edit_subset.subset_state is not state1
        assert d1.edit_subset.subset_state is not state2

    def test_apply_roi_doesnt_add_if_any_selection(self):
        d1 = self.add_data_and_attributes()
        d2 = self.add_data()
        d1.edit_subset = None
        d2.edit_subset = d2.new_subset()
        ct = len(d1.subsets)
        roi = RectangularROI()
        roi.update_limits(*self.roi_limits)
        x, y = self.roi_points
        self.client.apply_roi(roi)
        assert len(d1.subsets) == ct

    def test_subsets_drawn_over_data(self):
        data = self.add_data_and_attributes()
        subset = data.new_subset()
        assert self.is_first_in_front(subset, data)

    def test_log_sticky(self):
        self.add_data_and_attributes()
        self.assert_logs(False, False)

        self.client.xlog = True
        self.client.ylog = True
        self.assert_logs(True, True)

        self.client.xatt = self.ids[1]
        self.client.yatt = self.ids[0]
        self.assert_logs(True, True)

    def test_log_ticks(self):
        # regression test for 354
        self.add_data_and_attributes()
        self.assert_logs(False, False)

        self.client.xlog = True

        self.client.yatt = self.ids[0]

        self.assert_logs(True, False)
        assert not isinstance(self.client.axes.yaxis.get_major_locator(),
                              LogLocator)

    def assert_logs(self, xlog, ylog):
        ax = self.client.axes
        assert ax.get_xscale() == ('log' if xlog else 'linear')
        assert ax.get_yscale() == ('log' if ylog else 'linear')

    def test_flip_sticky(self):
        self.add_data_and_attributes()
        self.client.xflip = True
        self.assert_flips(True, False)
        self.client.xatt = self.ids[1]
        self.assert_flips(True, False)
        self.client.xatt = self.ids[0]
        self.assert_flips(True, False)

    def test_visibility_sticky(self):
        data = self.add_data_and_attributes()
        roi = RectangularROI()
        roi.update_limits(*self.roi_limits)
        assert self.client.is_visible(data.edit_subset)
        self.client.apply_roi(roi)
        self.client.set_visible(data.edit_subset, False)
        assert not self.client.is_visible(data.edit_subset)
        self.client.apply_roi(roi)
        assert not self.client.is_visible(data.edit_subset)

    def test_2d_data(self):
        """Should be abple to plot 2d data"""
        data = self.setup_2d_data()
        assert self.layer_data_correct(data, [1, 2, 3, 4], [2, 4, 6, 8])

    def test_2d_data_limits_with_subset(self):
        """visible limits should work with subsets and 2d data"""
        d = self.setup_2d_data()
        state = d.id['x'] > 2
        s = d.new_subset()
        s.subset_state = state
        assert self.client._visible_limits(0) == (1, 4)
        assert self.client._visible_limits(1) == (2, 8)

    def test_limits_nans(self):
        d = Data()
        x = Component(np.array([[1, 2], [np.nan, 4]]))
        y = Component(np.array([[2, 4], [np.nan, 8]]))
        xid = d.add_component(x, 'x')
        yid = d.add_component(y, 'y')
        self.collect.append(d)
        self.client.add_layer(d)
        self.client.xatt = xid
        self.client.yatt = yid

        assert self.client._visible_limits(0) == (1, 4)
        assert self.client._visible_limits(1) == (2, 8)

    def test_limits_inf(self):
        d = Data()
        x = Component(np.array([[1, 2], [np.infty, 4]]))
        y = Component(np.array([[2, 4], [-np.infty, 8]]))
        xid = d.add_component(x, 'x')
        yid = d.add_component(y, 'y')
        self.collect.append(d)
        self.client.add_layer(d)
        self.client.xatt = xid
        self.client.yatt = yid

        assert self.client._visible_limits(0) == (1, 4)
        assert self.client._visible_limits(1) == (2, 8)

    def test_xlog_relimits_if_negative(self):
        self.add_data_and_attributes()
        self.client.xflip = False
        self.client.xlog = False

        self.client.axes.set_xlim(-5, 5)
        self.client.xlog = True
        assert self.client.axes.get_xlim()[0] > .9

    def test_ylog_relimits_if_negative(self):
        self.add_data_and_attributes()
        self.client.yflip = False
        self.client.ylog = False
        self.client.axes.set_ylim(-5, 5)

        self.client.ylog = True
        assert self.client.axes.get_ylim()[0] > .9

    def test_subset_added_only_if_data_layer_present(self):
        self.collect.append(self.data[0])
        assert self.data[0] not in self.client.artists
        s = self.data[0].new_subset()
        assert s not in self.client.artists

    def test_pull_properties(self):
        ax = self.client.axes
        ax.set_xlim(6, 5)
        ax.set_ylim(8, 7)
        ax.set_xscale('log')
        ax.set_yscale('log')

        self.client._pull_properties()
        self.assert_properties_correct()

    def test_rescaled_on_init(self):
        layer = self.setup_2d_data()
        self.assert_layer_inside_limits(layer)

    def test_set_limits(self):
        self.client.xmin = 3
        self.client.xmax = 4
        self.client.ymin = 5
        self.client.ymax = 6
        ax = self.client.axes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        assert xlim[0] == self.client.xmin
        assert xlim[1] == self.client.xmax
        assert ylim[0] == self.client.ymin
        assert ylim[1] == self.client.ymax

    def test_ignore_duplicate_updates(self):
        """Need not create new artist on every draw. Enforce this"""
        layer = self.setup_2d_data()

        m = MagicMock()
        self.client.artists[layer][0].clear = m

        self.client._update_layer(layer)
        ct0 = m.call_count

        self.client._update_layer(layer)
        ct1 = m.call_count

        assert ct1 == ct0

    def test_range_rois_preserved(self):
        data = self.add_data_and_attributes()
        assert self.client.xatt is not self.client.yatt

        roi = XRangeROI()
        roi.set_range(1, 2)
        self.client.apply_roi(roi)
        assert isinstance(data.edit_subset.subset_state,
                          RangeSubsetState)
        assert data.edit_subset.subset_state.att == self.client.xatt

        roi = RectangularROI()
        roi = YRangeROI()
        roi.set_range(1, 2)
        self.client.apply_roi(roi)
        assert data.edit_subset.subset_state.att == self.client.yatt

    def test_component_replaced(self):
        # regression test for #508
        data = self.add_data_and_attributes()
        test = ComponentID('test')
        data.update_id(self.client.xatt, test)
        assert self.client.xatt is test


class TestCategoricalScatterClient(TestScatterClient):

    def setup_method(self, method):
        self.data = example_data.test_categorical_data()
        self.ids = [self.data[0].find_component_id('x1'),
                    self.data[0].find_component_id('y1'),
                    self.data[1].find_component_id('x2'),
                    self.data[1].find_component_id('y2')]
        self.roi_limits = (0.5, 0.5, 4, 4)
        self.roi_points = (np.array([1]), np.array([3]))
        self.collect = DataCollection()
        self.hub = self.collect.hub

        FIGURE.clf()
        axes = FIGURE.add_subplot(111)
        self.client = ScatterClient(self.collect, axes=axes)

        self.connect()

    def test_get_category_tick(self):

        self.add_data()
        self.client.xatt = self.ids[0]
        self.client.yatt = self.ids[0]
        axes = self.client.axes
        xformat = axes.xaxis.get_major_formatter()
        yformat = axes.yaxis.get_major_formatter()

        xlabels = [xformat.format_data(pos) for pos in range(2)]
        ylabels = [yformat.format_data(pos) for pos in range(2)]
        assert xlabels == ['a', 'b']
        assert ylabels == ['a', 'b']

    def test_axis_labels_sync_with_setters(self):
        layer = self.add_data()
        self.client.xatt = self.ids[0]
        assert self.client.axes.get_xlabel() == self.ids[0].label
        self.client.yatt = self.ids[1]
        assert self.client.axes.get_ylabel() == self.ids[1].label

    def test_jitter_with_setter_change(self):

        grab_data = lambda client: client.data[0][client.xatt].copy()
        layer = self.add_data()
        self.client.xatt = self.ids[0]
        self.client.yatt = self.ids[1]
        orig_data = grab_data(self.client)
        self.client.jitter = None
        np.testing.assert_equal(orig_data, grab_data(self.client))
        self.client.jitter = 'uniform'
        delta = np.abs(orig_data - grab_data(self.client))
        assert np.all((delta > 0) & (delta < 1))
        self.client.jitter = None
        np.testing.assert_equal(orig_data, grab_data(self.client))

    def test_ticks_go_back_after_changing(self):
        """ If you change to a categorical axis and then change back
        to a numeric, the axis ticks should fix themselves properly.
        """
        data = Data()
        data.add_component(Component(np.arange(100)), 'y')
        data.add_component(
            CategoricalComponent(['a'] * 50 + ['b'] * 50), 'xcat')
        data.add_component(Component(2 * np.arange(100)), 'xcont')

        self.add_data(data=data)
        self.client.yatt = data.find_component_id('y')
        self.client.xatt = data.find_component_id('xcat')
        self.check_ticks(self.client.axes.xaxis, False, True)
        self.check_ticks(self.client.axes.yaxis, False, False)

        self.client.xatt = data.find_component_id('xcont')
        self.check_ticks(self.client.axes.yaxis, False, False)
        self.check_ticks(self.client.axes.xaxis, False, False)

    def test_high_cardinatility_timing(self):

        card = 50000
        data = Data()
        card_data = [str(num) for num in range(card)]
        data.add_component(Component(np.arange(card * 5)), 'y')
        data.add_component(
            CategoricalComponent(np.repeat([card_data], 5)), 'xcat')
        self.add_data(data)
        comp = data.find_component_id('xcat')
        timer_func = partial(self.client._set_xydata, 'x', comp)

        timer = timeit(timer_func, number=1)
        assert timer < 3  # this is set for Travis speed

    def test_apply_roi(self):
        data = self.add_data_and_attributes()
        roi = RectangularROI()
        roi.update_limits(*self.roi_limits)
        x, y = self.roi_points
        self.client.apply_roi(roi)

    def test_range_rois_preserved(self):
        data = self.add_data_and_attributes()
        assert self.client.xatt is not self.client.yatt

        roi = XRangeROI()
        roi.set_range(1, 2)
        self.client.apply_roi(roi)
        assert isinstance(data.edit_subset.subset_state,
                          CategoricalROISubsetState)
        assert data.edit_subset.subset_state.att == self.client.xatt

        roi = YRangeROI()
        roi.set_range(1, 2)
        self.client.apply_roi(roi)
        assert isinstance(data.edit_subset.subset_state,
                          RangeSubsetState)
        assert data.edit_subset.subset_state.att == self.client.yatt
        roi = RectangularROI(xmin=1, xmax=2, ymin=1, ymax=2)

        self.client.apply_roi(roi)
        assert isinstance(data.edit_subset.subset_state,
                          AndState)

    @pytest.mark.parametrize(('roi_limits', 'mask'), [((0, -0.1, 10, 0.1), [0, 0, 0]),
                                                      ((0, 0.9, 10, 1.1), [1, 0, 0]),
                                                      ((0, 1.9, 10, 2.1), [0, 1, 0]),
                                                      ((0, 2.9, 10, 3.1), [0, 0, 1]),
                                                      ((0, 0.9, 10, 3.1), [1, 1, 1]),
                                                      ((-0.1, -1, 0.1, 5), [1, 1, 0]),
                                                      ((0.9, -1, 1.1, 5), [0, 0, 1]),
                                                      ((-0.1, 0.9, 1.1, 3.1), [1, 1, 1])])
    def test_apply_roi_results(self, roi_limits, mask):
        # Regression test for glue-viz/glue#718
        data = self.add_data_and_attributes()
        roi = RectangularROI()
        roi.update_limits(*roi_limits)
        x, y = self.roi_points
        self.client.apply_roi(roi)
        np.testing.assert_equal(data.edit_subset.to_mask(), mask)

    # REMOVED TESTS
    def test_invalid_plot(self):
        """ This fails because the axis ticks shouldn't reset after
        invalid plot. Current testing logic can't cope with this."""
        pass

    def test_redraw_called_on_invalid_plot(self):
        """ This fails because the axis ticks shouldn't reset after
        invalid plot. Current testing logic can't cope with this."""
        pass

    def test_xlog_relimits_if_negative(self):
        """ Log-based tests don't make sense here."""
        pass

    def test_log_sticky(self):
        """ Log-based tests don't make sense here."""
        pass

    def test_logs(self):
        """ Log-based tests don't make sense here."""
        pass
