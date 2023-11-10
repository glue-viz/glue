import numpy as np

from glue.tests.visual.helpers import visual_test
from glue.viewers.image.viewer import SimpleImageViewer
from glue.core.application_base import Application
from glue.core.data import Data
from glue.core.link_helpers import LinkSame
from glue.core.data_region import RegionData

from shapely.geometry import Polygon, MultiPolygon


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

    app.data_collection.new_subset_group(label='subset1', subset_state=region_data.id['values'] > 2)

    return viewer.figure
