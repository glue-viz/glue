from __future__ import absolute_import, division, print_function

from glue.core import Data
from glue.utils import renderless_figure

from ..layer_artist import ScatterLayerArtist

FIGURE = renderless_figure()


class TestScatterArtist(object):

    def setup_method(self, method):
        self.ax = FIGURE.add_subplot(111)

    def test_emphasis_compatible_with_data(self):
        # regression test for issue 249
        d = Data(x=[1, 2, 3])
        s = ScatterLayerArtist(d, self.ax)
        s.xatt = d.id['x']
        s.yatt = d.id['x']
        s.emphasis = d.id['x'] > 1

        s.update()
