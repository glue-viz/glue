from mock import MagicMock

from glue.core import Data, DataCollection
from ..layer_artist import HistogramLayerArtist
from ..state import HistogramViewerState

class TestHistogramLayerArtist(object):

    def setup_method(self):
        self.viewer_state = HistogramViewerState()
        ax = MagicMock()
        self.data = Data(x=[1, 2, 3])
        self.subset = self.data.new_subset()
        self.subset.subset_state = self.data.id['x'] > 1
        dc = DataCollection([self.data])
        # TODO: The following line shouldn't be needed
        self.viewer_state.data_collection = dc
        self.artist = HistogramLayerArtist(self.subset, ax, self.viewer_state)
        self.laye_state = self.artist.layer_state

    def setup_hist_calc_counter(self):
        m = MagicMock()
        self.artist._calculate_histogram = m
        return m

    def setup_hist_scale_counter(self):
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
        self.viewer_state.hist_x_min -= 1
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 2

        # hi
        self.viewer_state.hist_x_max -= 1
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 3

        # nbins
        self.viewer_state.hist_n_bin += 1
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 4

        # xlog
        self.viewer_state.log_x ^= True
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 5

        # ylog -- no call
        self.viewer_state.log_y ^= True
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 5

        # cumulative -- no call
        self.viewer_state.cumulative ^= True
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 5

        # normed -- no call
        self.viewer_state.normalize ^= True
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 5

        # subset style -- no call
        self.subset.style.color = '#00ff00'
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 5

        # subset state
        self.subset.subset_state = self.data.id['x'] > 10
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
        self.viewer_state.hist_x_min -= 1
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 2

        # hi
        self.viewer_state.hist_x_max -= 1
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 3

        # nbins
        self.viewer_state.hist_n_bin += 1
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 4

        # xlog
        self.viewer_state.log_x ^= True
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 5

        # ylog
        self.viewer_state.log_y ^= True
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 6

        # cumulative
        self.viewer_state.cumulative ^= True
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 7

        # normed
        self.viewer_state.normalize ^= True
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 8

        # subset state
        self.subset.subset_state = self.data.id['x'] > 10
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 9

        # subset style -- no call
        self.subset.style.color = '#00ff00'
        self.artist.update()
        self.artist.update()
        assert ct.call_count == 9
