import pytest
import numpy as np
from shapely.geometry import MultiPolygon, Polygon, Point, LineString

from glue.core.data import RegionData
from glue.core.component import (Component, ExtendedComponent)
from glue.core.component_id import ComponentID
from glue.app.qt import GlueApplication
from glue.core.state import GlueUnSerializer


def set_up_extended_comp(shapely_array, cen_x_name, cen_y_name=None):
    representative_points = [s.representative_point() for s in shapely_array]
    cen_x_id = ComponentID(cen_x_name)
    center_x = Component(np.array([s.x for s in representative_points]))

    if cen_y_name is not None:
        cen_y_id = ComponentID(cen_y_name)
        center_y = Component(np.array([s.y for s in representative_points]))
        extend_comp = ExtendedComponent(shapely_array, parent_component_ids=[cen_x_id, cen_y_id])
    else:
        cen_y_id = None
        center_y = []
        extend_comp = ExtendedComponent(shapely_array, parent_component_ids=[cen_x_id])
    return (extend_comp, cen_x_id, cen_y_id, center_x, center_y)


def set_up_region_data(shapely_array, cen_x_name, cen_y_name=None, **kwargs):
    extend_comp, cen_x_id, cen_y_id, center_x, center_y = set_up_extended_comp(shapely_array, cen_x_name, cen_y_name)
    region_data = RegionData(regions=extend_comp, **kwargs)
    # This is one way to add these components is a way that we can easily check their IDs
    region_data.add_component(center_x, cen_x_name)
    if cen_y_name is not None:
        region_data.add_component(center_y, cen_y_name)
    return (region_data, cen_x_id, cen_y_id)


class TestRegionData:

    def setup_class(self):

        poly_1 = Polygon([(20, 20), (60, 20), (60, 40), (20, 40)])
        poly_2 = Polygon([(60, 50), (60, 70), (80, 70), (80, 50)])
        poly_3 = Polygon([(10, 10), (15, 10), (15, 15), (10, 15)])
        poly_4 = Polygon([(10, 20), (15, 20), (15, 30), (10, 30), (12, 25)])

        poly_3_4 = MultiPolygon([poly_3, poly_4])

        polys = np.array([poly_1, poly_2, poly_3_4])
        self.polys_2d, self.x_id, self.y_id = set_up_region_data(polys, 'x', 'y')

        circle_1 = Point(1.0, 0.0).buffer(1)
        circle_2 = Point(2.0, 3.0).buffer(2)

        circles = np.array([circle_1, circle_2])
        self.circles, self.x_id_circles, self.y_id_circles = set_up_region_data(circles, 'xx', 'yy')

        range_1 = LineString([[1.0, 0.0], [3.0, 0.0]])
        range_2 = LineString([[5.0, 0.0], [9.0, 0.0]])

        ranges = np.array([range_1, range_2])
        self.ranges, self.x_id_ranges, self.y_id_ranges = set_up_region_data(ranges, 'x')

    def test_basic_properties(self):
        assert self.polys_2d.ext_x == self.x_id
        assert self.polys_2d.ext_y == self.y_id
        assert len(self.polys_2d._extended_component_ids) == 1
        assert self.polys_2d.ndim == 1  # The data array is 1D, although the extended component describes 2D shapes

        assert self.circles.ext_x == self.x_id_circles
        assert self.circles.ext_y == self.y_id_circles
        assert len(self.circles._extended_component_ids) == 1
        assert self.circles.ndim == 1  # The data array is 1D, although the extended component describes 2D shapes

        assert self.ranges.ext_x == self.x_id_ranges
        assert self.ranges.ext_y is None
        assert len(self.ranges._extended_component_ids) == 1
        assert self.ranges.ndim == 1

    def test_error_on_wrong_num_extended_components(self):

        circle_1 = Point(3.0, 4.0).buffer(1)
        circle_2 = Point(2.0, 6.0).buffer(2)

        more_circles = np.array([circle_1, circle_2])

        extend_comp, cen_x_id, cen_y_id, center_x, center_y = set_up_extended_comp(more_circles, 'x2', 'y2')
        self.circles.add_component(center_x, 'x2')
        self.circles.add_component(center_y, 'y2')
        with pytest.raises(ValueError):
            self.circles.add_component(extend_comp, 'new_extended')

    def test_save_restore(self, tmpdir):

        self.app = GlueApplication()
        self.session = self.app.session
        self.hub = self.session.hub
        self.data_collection = self.session.data_collection

        self.data_collection.append(self.polys_2d)
        assert len(self.data_collection) == 1

        filename = tmpdir.join("test_regiondata_save_restore_session.glu").strpath
        self.session.application.save_session(filename)
        with open(filename, "r") as f:
            session = f.read()

        state = GlueUnSerializer.loads(session)
        ga = state.object("__main__")
        dc = ga.session.data_collection
        assert len(dc) == 1
