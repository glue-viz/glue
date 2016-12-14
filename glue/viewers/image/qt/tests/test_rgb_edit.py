from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt

from glue.core import Data
from glue.viewers.image.layer_artist import RGBImageLayerArtist

from ..rgb_edit import RGBEdit

class TestRGBEdit(object):

    def setup_method(self, method):
        d = Data()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.artist = RGBImageLayerArtist(d, self.ax)
        self.w = RGBEdit(artist=self.artist)

    def teardown_method(self, method):
        plt.close(self.fig)

    def test_update_visible(self):
        for color in ['red', 'green', 'blue']:
            state = self.artist.layer_visible[color]
            self.w.vis[color].click()
            assert self.artist.layer_visible[color] != state

    def test_update_current(self):
        for color in ['red', 'green', 'blue']:
            self.w.current[color].click()
            assert self.artist.contrast_layer == color

    def test_disable_current(self):

        # If the current layer is set to not be visible, the current layer
        # needs to automatically change.

        self.w.current['red'].click()
        self.w.vis['red'].click()

        assert not self.w.current['red'].isEnabled()
        assert self.w.current['green'].isEnabled()
        assert self.w.current['blue'].isEnabled()

        assert not self.w.current['red'].isChecked()
        assert self.w.current['green'].isChecked()
        assert not self.w.current['blue'].isChecked()

        assert self.w.rgb_visible == (False, True, True)

        self.w.vis['red'].click()

        assert self.w.current['red'].isEnabled()
        assert self.w.current['green'].isEnabled()
        assert self.w.current['blue'].isEnabled()

        assert not self.w.current['red'].isChecked()
        assert self.w.current['green'].isChecked()
        assert not self.w.current['blue'].isChecked()

        assert self.w.rgb_visible == (True, True, True)

        self.w.vis['blue'].click()

        assert self.w.current['red'].isEnabled()
        assert self.w.current['green'].isEnabled()
        assert not self.w.current['blue'].isEnabled()

        assert not self.w.current['red'].isChecked()
        assert self.w.current['green'].isChecked()
        assert not self.w.current['blue'].isChecked()

        assert self.w.rgb_visible == (True, True, False)

        self.w.vis['green'].click()

        assert self.w.current['red'].isEnabled()
        assert not self.w.current['green'].isEnabled()
        assert not self.w.current['blue'].isEnabled()

        assert self.w.current['red'].isChecked()
        assert not self.w.current['green'].isChecked()
        assert not self.w.current['blue'].isChecked()

        assert self.w.rgb_visible == (True, False, False)

        self.w.vis['red'].click()

        assert not self.w.current['red'].isEnabled()
        assert not self.w.current['green'].isEnabled()
        assert not self.w.current['blue'].isEnabled()

        assert self.w.current['red'].isChecked()
        assert not self.w.current['green'].isChecked()
        assert not self.w.current['blue'].isChecked()

        assert self.w.rgb_visible == (False, False, False)

        self.w.vis['green'].click()

        assert not self.w.current['red'].isEnabled()
        assert self.w.current['green'].isEnabled()
        assert not self.w.current['blue'].isEnabled()

        assert not self.w.current['red'].isChecked()
        assert self.w.current['green'].isChecked()
        assert not self.w.current['blue'].isChecked()

        assert self.w.rgb_visible == (False, True, False)
