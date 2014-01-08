import pytest
import numpy as np

from ...core import Data, DataCollection
from ...qt.glue_application import GlueApplication
from ...qt.widgets import ScatterWidget, ImageWidget, HistogramWidget
from ..export_plotly import build_plotly_call

try:
    import plotly
    PLOTLY_INSTALLED = True
except ImportError:
    PLOTLY_INSTALLED = False


pytest.mark.skipif('not PLOTLY_INSTALLED')


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

        expected = dict(type='scatter', mode='markers', name=d.label,
                        marker=dict(size=6, color='rgba(255, 0, 0, 0.4)',
                                    symbol='circle'))
        for k, v in expected.items():
            assert args[0][k] == v

        np.testing.assert_array_equal(args[0]['x'], d['y'])
        np.testing.assert_array_equal(args[0]['y'], d['x'])

        assert 'layout' in kwargs
        layout = kwargs['layout']
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

        # check that subset is on Top
        assert len(args) == 2
        assert args[0]['name'] == 'data'
        assert args[1]['name'] == 'subset'

    def test_axes(self):
        app = self.app
        v = app.new_data_viewer(ScatterWidget, data=self.data)
        v.xlog = True
        v.xmin = 10
        v.xmax = 100

        v.ylog = False
        v.ymin = 2
        v.ymax = 4

        args, kwargs = build_plotly_call(app)

        xaxis = dict(type='log', rangemode='normal',
                     range=[1, 2], title='y')
        yaxis = dict(type='linear', rangemode='normal',
                     range=[2, 4], title='x')
        layout = kwargs['layout']
        assert layout['xaxis'] == xaxis
        assert layout['yaxis'] == yaxis

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
                color='rgba(0, 0, 0, 0.5)'
            ),
        )
        for k in expected:
            assert expected[k] == args[0][k]
        assert kwargs['layout']['barmode'] == 'overlay'

    def test_2plot(self):
        app = self.app
        d = self.data
        v = app.new_data_viewer(HistogramWidget, data=d)
        v2 = app.new_data_viewer(ScatterWidget, data=d)

        args, kwargs = build_plotly_call(app)

        assert len(args) == 2
        assert 'xaxis' not in args[0] and 'yaxis' not in args[0]
        assert args[1]['xaxis'] == 'x2'
        assert args[1]['yaxis'] == 'y2'

        layout = kwargs['layout']
        assert layout['xaxis']['domain'] == [0, .45]
        assert layout['xaxis2']['domain'] == [.55, 1]
        assert layout['yaxis2']['anchor'] == 'x2'

    def test_can_multiplot(self):
        # check that no errors are raised with 2-4 plots
        app = self.app
        d = self.data
        for i in range(2, 5):
            app.new_data_viewer(HistogramWidget, data=d)
            args, kwargs = build_plotly_call(app)

    def test_4plot(self):
        app = self.app
        d = self.data
        v = [app.new_data_viewer(HistogramWidget, data=d) for _ in range(4)]

        args, kwargs = build_plotly_call(app)

        assert len(args) == 4
        assert 'xaxis' not in args[0] and 'yaxis' not in args[0]
        assert args[1]['xaxis'] == 'x2'
        assert args[1]['yaxis'] == 'y2'
        assert args[2]['xaxis'] == 'x3'
        assert args[2]['yaxis'] == 'y3'
        assert args[3]['xaxis'] == 'x4'
        assert args[3]['yaxis'] == 'y4'

        layout = kwargs['layout']
        assert layout['xaxis']['domain'] == [0, .45]
        assert layout['yaxis']['domain'] == [0, .45]
        assert layout['xaxis2']['domain'] == [.55, 1]
        assert layout['yaxis2']['domain'] == [0, 0.45]
        assert layout['yaxis2']['anchor'] == 'x2'
        assert layout['xaxis3']['domain'] == [0, 0.45]
        assert layout['xaxis3']['anchor'] == 'y3'
        assert layout['yaxis3']['domain'] == [0.55, 1]
        assert layout['xaxis4']['anchor'] == 'y4'
        assert layout['yaxis4']['domain'] == [0.55, 1]
        assert layout['yaxis4']['anchor'] == 'x4'
