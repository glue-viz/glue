import numpy as np
import pytest
from numpy.testing import assert_allclose

from glue.core.application_base import Application
from glue.core.data import Data
from glue.core.link_helpers import LinkSame
from glue.core.tests.test_state import clone
from glue.viewers.scatter.viewer import SimpleScatterViewer
from glue.viewers.profile.viewer import SimpleProfileViewer
from glue.viewers.histogram.viewer import SimpleHistogramViewer
from glue.viewers.image.viewer import SimpleImageViewer
from glue.viewers.matplotlib.line_layers import (VerticalLineLayerArtist,
                                                 HorizontalLineLayerArtist,
                                                 add_vertical_lines,
                                                 add_horizontal_lines)
from glue.viewers.matplotlib.tests.test_python_export import BaseTestExportPython


def make_lines_app(main_data, link_cid):
    app = Application()
    lines = Data(position=[1., 3., 2., 3.], label='lines')
    app.data_collection.append(main_data)
    app.data_collection.append(lines)
    app.data_collection.add_link(LinkSame(lines.id['position'], link_cid))
    return app, lines


def assert_positions(artist, expected, horizontal=False):
    segments = artist.line_collection.get_segments()
    index = 1 if horizontal else 0
    assert_allclose([s[0][index] for s in segments], expected)


def test_scatter_viewer():

    data = Data(x=[1., 2., 3., 4.], y=[4., 5., 6., 7.], label='data')
    app, lines = make_lines_app(data, data.id['x'])

    viewer = app.new_data_viewer(SimpleScatterViewer)
    viewer.add_data(data)

    artist = add_vertical_lines(viewer, lines)

    assert isinstance(artist, VerticalLineLayerArtist)
    assert artist.enabled
    # Values are unique-d and sorted
    assert_positions(artist, [1, 2, 3])
    assert artist.line_collection.get_transform() is viewer.axes.get_xaxis_transform()

    # If the x attribute changes to one that cannot be resolved for the
    # layer, the layer is disabled, and re-enabled when changing back
    viewer.state.x_att = data.id['y']
    assert not artist.enabled
    viewer.state.x_att = data.id['x']
    assert artist.enabled
    assert_positions(artist, [1, 2, 3])


def test_scatter_viewer_horizontal():

    data = Data(x=[1., 2., 3., 4.], y=[4., 5., 6., 7.], label='data')
    app, lines = make_lines_app(data, data.id['y'])

    viewer = app.new_data_viewer(SimpleScatterViewer)
    viewer.add_data(data)

    artist = add_horizontal_lines(viewer, lines)

    assert isinstance(artist, HorizontalLineLayerArtist)
    assert artist.enabled
    assert_positions(artist, [1, 2, 3], horizontal=True)
    assert artist.line_collection.get_transform() is viewer.axes.get_yaxis_transform()


def test_profile_viewer():

    spectrum = Data(flux=np.random.random(10), label='spectrum')
    app, lines = make_lines_app(spectrum, spectrum.pixel_component_ids[0])

    viewer = app.new_data_viewer(SimpleProfileViewer)
    viewer.add_data(spectrum)

    artist = add_vertical_lines(viewer, lines)

    assert artist.enabled
    assert_positions(artist, [1, 2, 3])

    # Subsets created after the fact follow the parent dataset and are also
    # shown as vertical lines. Note that the subset group is global, so the
    # spectrum subset is added too, as a regular profile layer.
    app.data_collection.new_subset_group(label='high', subset_state=lines.id['position'] > 1.5)

    subset_artists = [layer for layer in viewer.layers
                      if isinstance(layer, VerticalLineLayerArtist)
                      and layer.layer is not lines]
    assert len(subset_artists) == 1
    assert subset_artists[0].layer.data is lines
    assert subset_artists[0].enabled
    assert_positions(subset_artists[0], [2, 3])

    # Removing the data removes the vertical line layers
    viewer.remove_data(lines)
    assert not any(isinstance(layer, VerticalLineLayerArtist) for layer in viewer.layers)


def test_histogram_viewer():

    data = Data(values=[1., 1., 2., 3., 5., 8.], label='data')
    app, lines = make_lines_app(data, data.id['values'])

    viewer = app.new_data_viewer(SimpleHistogramViewer)
    viewer.add_data(data)

    artist = add_vertical_lines(viewer, lines)

    assert artist.enabled
    assert_positions(artist, [1, 2, 3])


def test_image_viewer():

    image = Data(intensity=np.random.random((5, 5)), label='image')
    app, lines = make_lines_app(image, image.pixel_component_ids[1])

    viewer = app.new_data_viewer(SimpleImageViewer)
    viewer.add_data(image)

    artist = add_vertical_lines(viewer, lines)

    assert artist.enabled
    assert_positions(artist, [1, 2, 3])


def test_image_viewer_horizontal():

    image = Data(intensity=np.random.random((5, 5)), label='image')
    app, lines = make_lines_app(image, image.pixel_component_ids[0])

    viewer = app.new_data_viewer(SimpleImageViewer)
    viewer.add_data(image)

    artist = add_horizontal_lines(viewer, lines)

    assert artist.enabled
    assert_positions(artist, [1, 2, 3], horizontal=True)


def test_horizontal_not_available():

    # The profile and histogram viewers have no y axis attribute so
    # horizontal lines are not available

    spectrum = Data(flux=np.random.random(10), label='spectrum')
    app, lines = make_lines_app(spectrum, spectrum.pixel_component_ids[0])

    for viewer_cls in (SimpleProfileViewer, SimpleHistogramViewer):
        viewer = app.new_data_viewer(viewer_cls)
        viewer.add_data(spectrum)
        with pytest.raises(ValueError, match='does not define a y axis'):
            add_horizontal_lines(viewer, lines)


def test_existing_subsets_added():

    spectrum = Data(flux=np.random.random(10), label='spectrum')
    app, lines = make_lines_app(spectrum, spectrum.pixel_component_ids[0])

    app.data_collection.new_subset_group(label='high', subset_state=lines.id['position'] > 1.5)

    viewer = app.new_data_viewer(SimpleProfileViewer)
    viewer.add_data(spectrum)

    add_vertical_lines(viewer, lines)

    vline_artists = [layer for layer in viewer.layers
                     if isinstance(layer, VerticalLineLayerArtist)]
    assert len(vline_artists) == 2
    assert vline_artists[1].layer.data is lines
    assert_positions(vline_artists[1], [2, 3])


def test_styling():

    data = Data(x=[1., 2., 3.], y=[4., 5., 6.], label='data')
    app, lines = make_lines_app(data, data.id['x'])

    lines.style.color = '#ff0000'

    viewer = app.new_data_viewer(SimpleScatterViewer)
    viewer.add_data(data)
    artist = add_vertical_lines(viewer, lines)

    assert artist.state.color == '#ff0000'

    artist.state.linewidth = 3
    assert artist.line_collection.get_linewidth()[0] == 3

    artist.state.linestyle = 'dashed'

    artist.state.visible = False
    assert not artist.line_collection.get_visible()


def test_state_round_trip():

    # The line layer state can be cloned, which is what session saving relies
    # on (the full application round trip is tested in glue-qt, which is
    # where sessions including viewers are saved)

    data = Data(x=[1., 2., 3., 4.], y=[4., 5., 6., 7.], label='data')
    app, lines = make_lines_app(data, data.id['x'])

    viewer = app.new_data_viewer(SimpleScatterViewer)
    viewer.add_data(data)
    artist = add_vertical_lines(viewer, lines)
    artist.state.linewidth = 4
    artist.state.linestyle = 'dotted'

    state2 = clone(artist.state)
    assert state2.linewidth == 4
    assert state2.linestyle == 'dotted'


def test_many_lines():

    # All the lines are rendered as a single LineCollection so this should
    # remain fast even for many thousands of lines

    data = Data(x=np.arange(100000.), label='data')
    lines = Data(position=np.random.random(50000) * 1e5, label='lines')

    app = Application()
    app.data_collection.append(data)
    app.data_collection.append(lines)
    app.data_collection.add_link(LinkSame(lines.id['position'], data.id['x']))

    viewer = app.new_data_viewer(SimpleHistogramViewer)
    viewer.add_data(data)

    artist = add_vertical_lines(viewer, lines)

    assert artist.enabled
    assert len(artist.line_collection.get_segments()) == 50000
    assert len(artist.axes.collections) == 1


class TestExportPython(BaseTestExportPython):

    def setup_method(self, method):
        data = Data(x=[1., 2., 3., 4.], y=[4., 5., 6., 7.], label='data')
        self.app, self.lines = make_lines_app(data, data.id['x'])
        self.viewer = self.app.new_data_viewer(SimpleScatterViewer)
        self.viewer.add_data(data)
        self.viewer.state.legend.location = 'lower left'

    def teardown_method(self, method):
        self.viewer = None
        self.app = None

    def test_vertical(self, tmpdir):
        add_vertical_lines(self.viewer, self.lines)
        self.assert_same(tmpdir)

    def test_horizontal(self, tmpdir):
        self.app.data_collection.add_link(
            LinkSame(self.lines.id['position'],
                     self.app.data_collection['data'].id['y']))
        add_horizontal_lines(self.viewer, self.lines)
        self.assert_same(tmpdir)

    def test_styled(self, tmpdir):
        artist = add_vertical_lines(self.viewer, self.lines)
        artist.state.linewidth = 3
        artist.state.linestyle = 'dashed'
        artist.state.alpha = 0.5
        self.assert_same(tmpdir)
