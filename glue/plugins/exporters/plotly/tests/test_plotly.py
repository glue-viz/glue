from __future__ import absolute_import, division, print_function

import pytest
import numpy as np

from glue.config import settings
from glue.core import Data, DataCollection

pytest.importorskip('qtpy')

from glue.app.qt import GlueApplication
from glue.viewers.scatter.qt import ScatterViewer
from glue.viewers.histogram.qt import HistogramViewer

from ..export_plotly import build_plotly_call


class TestPlotly(object):

    def setup_method(self, method):
        d = Data(x=[1, 2, 3], y=[2, 3, 4], label='data')
        dc = DataCollection([d])
        self.app = GlueApplication(dc)
        self.data = d

    def teardown_method(self, method):
        self.app.close()
        self.app = None

    def test_scatter(self):

        d = self.data
        d.style.markersize = 6
        d.style.color = '#ff0000'
        d.style.alpha = .4

        viewer = self.app.new_data_viewer(ScatterViewer, data=d)
        viewer.state.x_att = d.id['y']
        viewer.state.y_att = d.id['x']

        args, kwargs = build_plotly_call(self.app)
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

        viewer.close()

    def test_scatter_subset(self):

        d = self.data
        s = d.new_subset(label='subset')
        s.subset_state = d.id['x'] > 1
        s.style.marker = 's'

        viewer = self.app.new_data_viewer(ScatterViewer, data=d)
        viewer.state.x_att = d.id['x']
        viewer.state.y_att = d.id['x']

        args, kwargs = build_plotly_call(self.app)
        data = args[0]['data']

        # check that subset is on Top
        assert len(data) == 2
        assert data[0]['name'] == 'data'
        assert data[1]['name'] == 'subset'

        viewer.close()

    def test_axes(self):

        viewer = self.app.new_data_viewer(ScatterViewer, data=self.data)

        viewer.state.x_log = True
        viewer.state.x_min = 10
        viewer.state.x_max = 100
        viewer.state.x_att = self.data.id['x']

        viewer.state.y_log = False
        viewer.state.y_min = 2
        viewer.state.y_max = 4
        viewer.state.y_att = self.data.id['y']

        args, kwargs = build_plotly_call(self.app)

        xaxis = dict(type='log', rangemode='normal',
                     range=[1, 2], title='x', zeroline=False)
        yaxis = dict(type='linear', rangemode='normal',
                     range=[2, 4], title='y', zeroline=False)
        layout = args[0]['layout']
        for k, v in layout['xaxis'].items():
            assert xaxis.get(k, v) == v
        for k, v in layout['yaxis'].items():
            assert yaxis.get(k, v) == v

        viewer.close()

    def test_histogram(self):

        d = self.data
        d.style.color = '#000000'

        viewer = self.app.new_data_viewer(HistogramViewer, data=d)
        viewer.component = d.id['y']
        viewer.state.hist_x_min = 0
        viewer.state.hist_x_max = 10
        viewer.state.hist_n_bin = 20

        args, kwargs = build_plotly_call(self.app)

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

        viewer.close()
