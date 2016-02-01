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