import numpy as np
from echo import delay_callback
from glue.tests.visual.helpers import visual_test
from glue.viewers.image.viewer import SimpleImageViewer
from glue.core.application_base import Application
from glue.core.data import Data
from glue.core.link_helpers import LinkSame
from glue.core.data_region import RegionData
from astropy.wcs import WCS

from shapely.geometry import Polygon, MultiPolygon, Point
import shapely


@visual_test
def test_simple_viewer():

    # Make sure the simple viewer can be instantiated

    data1 = Data(x=np.arange(6).reshape((2, 3)), label='data1')
    data2 = Data(y=2 * np.arange(6).reshape((2, 3)), label='data2')

    app = Application()
    app.data_collection.append(data1)
    app.data_collection.append(data2)

    viewer = app.new_data_viewer(SimpleImageViewer)
    viewer.add_data(data1)
    viewer.add_data(data2)

    app.data_collection.new_subset_group(label='subset1', subset_state=data1.pixel_component_ids[1] > 1.2)

    return viewer.figure


@visual_test
def test_region_layer():
    poly_1 = Polygon([(20, 20), (60, 20), (60, 40), (20, 40)])
    poly_2 = Polygon([(60, 50), (60, 70), (80, 70), (80, 50)])
    poly_3 = Polygon([(10, 10), (15, 10), (15, 15), (10, 15)])
    poly_4 = Polygon([(10, 20), (15, 20), (15, 30), (10, 30), (12, 25)])

    polygons = MultiPolygon([poly_3, poly_4])

    geoms = np.array([poly_1, poly_2, polygons])
    a_values = np.array([1, 2, 3])
    b_values = np.array([1, 2, 3])

    region_data = RegionData(regions=geoms, a=a_values, b=b_values)

    image_data = Data(x=np.arange(10000).reshape((100, 100)), label='data1')
    app = Application()
    app.data_collection.append(image_data)
    app.data_collection.append(region_data)

    viewer = app.new_data_viewer(SimpleImageViewer)
    viewer.add_data(image_data)
    viewer.add_data(region_data)

    # Link to region data components that are the not the x,y coordinates
    link1 = LinkSame(region_data.id['a'], image_data.pixel_component_ids[0])
    link2 = LinkSame(region_data.id['b'], image_data.pixel_component_ids[1])
    app.data_collection.add_link(link1)
    app.data_collection.add_link(link2)

    app.data_collection.remove_link(link1)
    app.data_collection.remove_link(link2)

    link1 = LinkSame(region_data.center_x_id, image_data.pixel_component_ids[0])
    link2 = LinkSame(region_data.center_y_id, image_data.pixel_component_ids[1])
    app.data_collection.add_link(link1)
    app.data_collection.add_link(link2)

    return viewer.figure


def test_region_layer_logic():
    poly_1 = Polygon([(20, 20), (60, 20), (60, 40), (20, 40)])
    poly_2 = Polygon([(60, 50), (60, 70), (80, 70), (80, 50)])
    poly_3 = Polygon([(10, 10), (15, 10), (15, 15), (10, 15)])
    poly_4 = Polygon([(10, 20), (15, 20), (15, 30), (10, 30), (12, 25)])

    polygons = MultiPolygon([poly_3, poly_4])

    geoms = np.array([poly_1, poly_2, polygons])
    a_values = np.array([1, 2, 3])
    b_values = np.array([1, 2, 3])

    region_data = RegionData(regions=geoms, a=a_values, b=b_values)

    image_data = Data(x=np.arange(10000).reshape((100, 100)), label='data1')
    app = Application()
    app.data_collection.append(image_data)
    app.data_collection.append(region_data)

    viewer = app.new_data_viewer(SimpleImageViewer)
    viewer.add_data(image_data)
    viewer.add_data(region_data)

    assert viewer.layers[0].enabled  # image
    assert not viewer.layers[1].enabled  # regions

    # Link to region data components that are the not the x,y coordinates
    link1 = LinkSame(region_data.id['a'], image_data.pixel_component_ids[0])
    link2 = LinkSame(region_data.id['b'], image_data.pixel_component_ids[1])
    app.data_collection.add_link(link1)
    app.data_collection.add_link(link2)

    assert viewer.layers[0].enabled  # image
    assert not viewer.layers[1].enabled  # regions

    app.data_collection.remove_link(link1)
    app.data_collection.remove_link(link2)

    link1 = LinkSame(region_data.center_x_id, image_data.pixel_component_ids[0])
    link2 = LinkSame(region_data.center_y_id, image_data.pixel_component_ids[1])
    app.data_collection.add_link(link1)
    app.data_collection.add_link(link2)

    assert viewer.layers[0].enabled  # image
    assert viewer.layers[1].enabled  # regions


@visual_test
def test_region_layer_flip():
    poly_1 = Polygon([(20, 20), (60, 20), (60, 40), (20, 40)])
    poly_2 = Polygon([(60, 50), (60, 70), (80, 70), (80, 50)])
    poly_3 = Polygon([(10, 10), (15, 10), (15, 15), (10, 15)])
    poly_4 = Polygon([(10, 20), (15, 20), (15, 30), (10, 30), (12, 25)])

    polygons = MultiPolygon([poly_3, poly_4])

    geoms = np.array([poly_1, poly_2, polygons])
    values = np.array([1, 2, 3])
    region_data = RegionData(regions=geoms, values=values)

    image_data = Data(x=np.arange(10000).reshape((100, 100)), label='data1')
    app = Application()
    app.data_collection.append(image_data)
    app.data_collection.append(region_data)

    link1 = LinkSame(region_data.center_x_id, image_data.pixel_component_ids[0])
    link2 = LinkSame(region_data.center_y_id, image_data.pixel_component_ids[1])
    app.data_collection.add_link(link1)
    app.data_collection.add_link(link2)

    viewer = app.new_data_viewer(SimpleImageViewer)
    viewer.add_data(image_data)
    viewer.add_data(region_data)

    # We need this delay callback here because, while this works in the QT GUI,
    # we need to make sure not to try and redraw the regions while we are flipping
    # the coordinates displayed.

    with delay_callback(viewer.state, 'x_att', 'y_att'):
        viewer.state.x_att = image_data.pixel_component_ids[0]
        viewer.state.y_att = image_data.pixel_component_ids[1]

    return viewer.figure


class TestWCSRegionDisplay(object):
    def setup_method(self, method):

        wcs1 = WCS(naxis=2)
        wcs1.wcs.ctype = 'RA---TAN', 'DEC--TAN'
        wcs1.wcs.crpix = 15, 15
        wcs1.wcs.cd = [[2, -1], [1, 2]]

        wcs1.wcs.set()

        np.random.seed(2)
        self.image1 = Data(label='image1', a=np.random.rand(30, 30), coords=wcs1)
        SHAPELY_ARRAY = np.array([Point(1.5, 2.5).buffer(4),
                                  Polygon([(10, 10), (10, 15), (20, 15), (20, 10)])])
        self.region_data = RegionData(label='My Regions',
                                      color=np.array(['red', 'blue']),
                                      area=shapely.area(SHAPELY_ARRAY),
                                      boundary=SHAPELY_ARRAY)
        self.application = Application()

        self.application.data_collection.append(self.image1)
        self.application.data_collection.append(self.region_data)

        self.viewer = self.application.new_data_viewer(SimpleImageViewer)

    def test_wcs_viewer_bad_link(self):
        self.viewer.add_data(self.image1)

        link1 = LinkSame(self.region_data.id['color'], self.image1.world_component_ids[1])
        link2 = LinkSame(self.region_data.id['area'], self.image1.world_component_ids[0])

        self.application.data_collection.add_link(link1)
        self.application.data_collection.add_link(link2)

        self.viewer.add_data(self.region_data)

        assert self.viewer.state._display_world is True
        assert len(self.viewer.state.layers) == 2
        assert self.viewer.layers[0].enabled
        assert not self.viewer.layers[1].enabled

    def test_wcs_viewer_good_link(self):
        self.viewer.add_data(self.image1)

        link1 = LinkSame(self.region_data.center_x_id, self.image1.world_component_ids[1])
        link2 = LinkSame(self.region_data.center_y_id, self.image1.world_component_ids[0])

        self.application.data_collection.add_link(link1)
        self.application.data_collection.add_link(link2)

        self.viewer.add_data(self.region_data)

        assert self.viewer.state._display_world is True
        assert len(self.viewer.state.layers) == 2
        assert self.viewer.layers[0].enabled
        assert self.viewer.layers[1].enabled

    @visual_test
    def test_wcs_viewer(self):
        self.viewer.add_data(self.image1)

        link1 = LinkSame(self.region_data.center_x_id, self.image1.world_component_ids[1])
        link2 = LinkSame(self.region_data.center_y_id, self.image1.world_component_ids[0])

        self.application.data_collection.add_link(link1)
        self.application.data_collection.add_link(link2)

        self.viewer.add_data(self.region_data)

        assert self.viewer.state._display_world is True
        assert len(self.viewer.state.layers) == 2
        assert self.viewer.layers[0].enabled
        assert self.viewer.layers[1].enabled

        return self.viewer.figure

    @visual_test
    def test_flipped_wcs_viewer(self):
        self.viewer.add_data(self.image1)

        link1 = LinkSame(self.region_data.center_x_id, self.image1.world_component_ids[1])
        link2 = LinkSame(self.region_data.center_y_id, self.image1.world_component_ids[0])

        self.application.data_collection.add_link(link1)
        self.application.data_collection.add_link(link2)

        self.viewer.add_data(self.region_data)
        original_path_patch = self.viewer.layers[1].region_collection.patches[1].get_path().vertices

        # Flip x,y in the viewer
        with delay_callback(self.viewer.state, 'x_att_world', 'y_att_world', 'x_att', 'y_att'):
            self.viewer.state.x_att_world = self.image1.world_component_ids[0]
            self.viewer.state.y_att_world = self.image1.world_component_ids[1]
            self.viewer.state.x_att = self.image1.pixel_component_ids[0]
            self.viewer.state.y_att = self.image1.pixel_component_ids[1]

        assert self.viewer.state._display_world is True
        assert len(self.viewer.state.layers) == 2
        assert self.viewer.layers[0].enabled
        assert self.viewer.layers[1].enabled
        new_path_patch = self.viewer.layers[1].region_collection.patches[1].get_path().vertices

        # Because we have flipped the viewer, the patches should have changed
        assert np.array_equal(original_path_patch, np.flip(new_path_patch, axis=1))

        return self.viewer.figure
