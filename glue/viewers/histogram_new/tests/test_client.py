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


class TestCategoricalHistogram(TestHistogramClient):

    def test_change_default_bins(self):
        self.client.add_layer(self.data)
        self.client.set_component(self.data.id['x'])
        assert self.client.nbins == 6

    def test_nbin_override_persists_over_attribute_change(self):
        # regression test for #398
        self.collect.append(self.data)
        self.client.add_layer(self.data)
        self.client.set_component(self.data.id['x'])
        self.client.nbins = 7
        self.client.set_component(self.data.id['y'])
        assert self.client.nbins == 7


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
