from collections import Counter

import sys

from glue.core import Data, DataCollection
from ..layer_artist import HistogramLayerArtist
from ..state import HistogramViewerState
from matplotlib import pyplot as plt


class CallCounter(object):

    def __init__(self):
        self._counter = Counter()

    def __call__(self, frame, event, arg):
        if event == 'call':
            self._counter[frame.f_code.co_name] += 1

    def __getitem__(self, item):
        return self._counter[item]


class TestHistogramLayerArtist(object):

    def setup_method(self, method):

        self.viewer_state = HistogramViewerState()

        ax = plt.subplot(1, 1, 1)

        self.data = Data(x=[1, 2, 3], y=[2, 3, 4])
        self.subset = self.data.new_subset()
        self.subset.subset_state = self.data.id['x'] > 1

        dc = DataCollection([self.data])

        # TODO: The following line shouldn't be needed
        self.viewer_state.data_collection = dc

        self.artist = HistogramLayerArtist(ax, self.viewer_state, layer=self.subset)
        self.layer_state = self.artist.state
        self.viewer_state.layers.append(self.layer_state)

        self.call_counter = CallCounter()
        sys.setprofile(self.call_counter)

    def teardown_method(self, method):
        self.artist.remove()
        sys.setprofile(None)

    def test_recalc_on_state_changes(self):

        assert self.call_counter['_calculate_histogram'] == 0
        assert self.call_counter['_update_artists'] == 0

        # attribute
        self.viewer_state.x_att = self.data.id['y']
        assert self.call_counter['_calculate_histogram'] == 1
        assert self.call_counter['_update_artists'] == 1

        # lo
        self.viewer_state.hist_x_min = -1
        assert self.call_counter['_calculate_histogram'] == 2
        assert self.call_counter['_update_artists'] == 2

        # hi
        self.viewer_state.hist_x_max = 5
        assert self.call_counter['_calculate_histogram'] == 3
        assert self.call_counter['_update_artists'] == 3

        # nbins
        self.viewer_state.hist_n_bin += 1
        assert self.call_counter['_calculate_histogram'] == 4
        assert self.call_counter['_update_artists'] == 4

        # xlog
        self.viewer_state.x_log ^= True
        assert self.call_counter['_calculate_histogram'] == 5
        assert self.call_counter['_update_artists'] == 5

        # TODO: find a way to determine whether the histogram calculation is
        # carried out since _calculate_histogram calls are no longer a good
        # way to find out (we now rely on state cache)

        # ylog -- no call
        self.viewer_state.y_log ^= True
        # assert self.call_counter['_calculate_histogram'] == 5
        assert self.call_counter['_update_artists'] == 6

        # cumulative -- no call
        self.viewer_state.cumulative ^= True
        # assert self.call_counter['_calculate_histogram'] == 5
        assert self.call_counter['_update_artists'] == 7

        # normed -- no call
        self.viewer_state.normalize ^= True
        # assert self.call_counter['_calculate_histogram'] == 5
        assert self.call_counter['_update_artists'] == 8

        # subset style -- no call
        self.subset.style.color = '#00ff00'
        # assert self.call_counter['_calculate_histogram'] == 5
        assert self.call_counter['_update_artists'] == 8

        # legend -- no call
        self.viewer_state.show_legend = True
        assert self.call_counter['_update_artists'] == 8
