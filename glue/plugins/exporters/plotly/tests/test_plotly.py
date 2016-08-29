from __future__ import absolute_import, division, print_function

import pytest
import numpy as np

from glue.config import settings
from glue.core import Data, DataCollection

pytest.importorskip('qtpy')

from glue.app.qt import GlueApplication
from glue.viewers.scatter.qt import ScatterWidget
from glue.viewers.histogram.qt import HistogramWidget

from ..export_plotly import build_plotly_call


class TestPlotly(object):

    def setup_method(self, method):
        d = Data(x=[1, 2, 3], y=[2, 3, 4], label='data')
        dc = DataCollection([d])
        self.app = GlueApplication(dc)
        self.data = d

    def test_scatter(self):
        app = self.app
        d = self.data
        d.style.markersize = 6
        d.style.color = '#ff0000'
        d.style.alpha = .4
        v = app.new_data_viewer(ScatterWidget, data=d)
        v.xatt = d.id['y']
        v.yatt = d.id['x']

        args, kwargs = build_plotly_call(app)
        data = args[0]['data'][0]

        expected = dict(type='scatter', mode='markers', name=d.label,
                        marker=dict(size=6, color='rgba(255, 0, 0, 0.4)',
                                    symbol='circle'))
        for k, v in expected.items():
            assert data[k] == v

        np.testing.assert_array_equal(data['x'], d['y'])
        np.testing.assert_array_equal(data['y'], d['x'])

        layout = args[0]['layout']
        assert layout['showlegend']

    def test_scatter_subset(self):
        app = self.app
        d = self.data
        s = d.new_subset(label='subset')
        s.subset_state = d.id['x'] > 1
        s.style.marker = 's'

        v = app.new_data_viewer(ScatterWidget, data=d)
        v.xatt = d.id['x']
        v.yatt = d.id['x']

        args, kwargs = build_plotly_call(app)
        data = args[0]['data']

        # check that subset is on Top
        assert len(data) == 2
        assert data[0]['name'] == 'data'
        assert data[1]['name'] == 'subset'

    def test_axes(self):

        app = self.app

        v = app.new_data_viewer(ScatterWidget, data=self.data)

        v.xlog = True
        v.xmin = 10
        v.xmax = 100
        v.xatt = self.data.id['x']

        v.ylog = False
        v.ymin = 2
        v.ymax = 4
        v.yatt = self.data.id['y']

        args, kwargs = build_plotly_call(app)

        xaxis = dict(type='log', rangemode='normal',
                     range=[1, 2], title='x', zeroline=False)
        yaxis = dict(type='linear', rangemode='normal',
                     range=[2, 4], title='y', zeroline=False)
        layout = args[0]['layout']
        for k, v in layout['xaxis'].items():
            assert xaxis.get(k, v) == v
        for k, v in layout['yaxis'].items():
            assert yaxis.get(k, v) == v

    def test_histogram(self):
        app = self.app
        d = self.data
        d.style.color = '#000000'
        v = app.new_data_viewer(HistogramWidget, data=d)
        v.component = d.id['y']
        v.xmin = 0
        v.xmax = 10
        v.bins = 20

        args, kwargs = build_plotly_call(app)

        expected = dict(
            name='data',
            type='bar',
            marker=dict(
                color='rgba(0, 0, 0, {0:0.1f})'.format(float(settings.DATA_ALPHA))
            ),
        )
        data = args[0]['data']
        for k in expected:
            assert expected[k] == data[0][k]
        assert args[0]['layout']['barmode'] == 'overlay'
