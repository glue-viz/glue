import pytest
from numpy.testing import assert_array_equal

import numpy as np
from shapely.geometry import MultiPolygon, Polygon, Point

from glue.core.data import Data, RegionData
from glue.core.component import ExtendedComponent
from glue.core.state import GlueUnSerializer
from glue.core.tests.test_application_base import MockApplication
from glue.core.link_helpers import LinkSame

poly_1 = Polygon([(20, 20), (60, 20), (60, 40), (20, 40)])
poly_2 = Polygon([(60, 50), (60, 70), (80, 70), (80, 50)])
poly_3 = Polygon([(10, 10), (15, 10), (15, 15), (10, 15)])
poly_4 = Polygon([(10, 20), (15, 20), (15, 30), (10, 30), (12, 25)])

polygons = MultiPolygon([poly_3, poly_4])
SHAPELY_POLYGON_ARRAY = np.array([poly_1, poly_2, polygons])

SHAPELY_CIRCLE_ARRAY = np.array([Point(0, 0).buffer(1), Point(1, 1).buffer(1)])
CENTER_X = [0, 1]
CENTER_Y = [0, 1]


class TestRegionData(object):

    def setup_method(self):
        self.region_data = RegionData(label='My Regions', boundary=SHAPELY_POLYGON_ARRAY)

        self.manual_region_data = RegionData(label='My Manual Regions')

        self.cen_x_id = self.manual_region_data.add_component(CENTER_X, label='Center [x]')
        self.cen_y_id = self.manual_region_data.add_component(CENTER_Y, label='Center [y]')
        self.extended_comp = ExtendedComponent(SHAPELY_CIRCLE_ARRAY, center_comp_ids=[self.cen_x_id, self.cen_y_id])
        self.manual_region_data.add_component(self.extended_comp, label='circles')

    def test_basic_properties_simple(self):
        assert self.region_data.label == 'My Regions'
        assert self.region_data.shape == SHAPELY_POLYGON_ARRAY.shape
        assert self.region_data.ndim == 1
        assert self.region_data.size == 3
        assert_array_equal(self.region_data['boundary'], SHAPELY_POLYGON_ARRAY)
        assert len(self.region_data.main_components) == 3
        component_labels = [cid.label for cid in self.region_data.main_components]
        assert 'boundary' in component_labels
        assert 'Center [x] for boundary' in component_labels
        assert 'Center [y] for boundary' in component_labels

    def test_basic_properties_manual(self):
        assert self.manual_region_data.label == 'My Manual Regions'
        assert self.manual_region_data.shape == np.asarray(SHAPELY_CIRCLE_ARRAY).shape
        assert self.manual_region_data.ndim == 1
        assert self.manual_region_data.size == 2
        assert_array_equal(self.manual_region_data['circles'], SHAPELY_CIRCLE_ARRAY)
        assert len(self.region_data.main_components) == 3
        component_labels = [cid.label for cid in self.manual_region_data.main_components]
        assert 'circles' in component_labels
        assert 'Center [x]' in component_labels
        assert 'Center [y]' in component_labels


class TestRegionDataSaveRestore(object):

    @pytest.fixture(autouse=True)
    def setup_method(self, tmpdir):
        app = MockApplication()
        geodata = RegionData(label='My Regions', regions=SHAPELY_POLYGON_ARRAY)
        catdata = Data(label='catdata', x=np.array([1, 2, 3, 4]), y=np.array([10, 20, 30, 40]))
        app.data_collection.append(geodata)
        app.data_collection.append(catdata)

        app.data_collection.add_link(LinkSame(geodata.id['Center [x] for regions'], catdata.id['x']))
        app.data_collection.add_link(LinkSame(geodata.id['Center [y] for regions'], catdata.id['y']))

        session_file = tmpdir.mkdir("session").join('test.glu')
        app.save_session(session_file)

        with open(session_file, "r") as f:
            session = f.read()

        state = GlueUnSerializer.loads(session)
        ga = state.object("__main__")
        dc = ga.session.data_collection

        self.reg_before = app.data_collection[0]
        self.cat_before = app.data_collection[1]

        self.reg_after = dc[0]
        self.cat_after = dc[1]

    def test_data_roundtrip(self):
        assert_array_equal(self.reg_before['regions'], self.reg_after['regions'])
        assert_array_equal(self.cat_before['x'], self.cat_after['x'])
        assert_array_equal(self.cat_before['y'], self.cat_after['y'])

    def test_component_ids_are_restored_correctly(self):
        for data in [self.reg_before, self.reg_after]:
            assert data.extended_component_id == data.id['regions']
            assert data.extended_component_id == data.id['regions']

            assert data.components[1] == data.get_component(data.components[3]).x
            assert data.components[2] == data.get_component(data.components[3]).y

    def test_links_still_work(self):
        for data in [(self.reg_before, self.cat_before), (self.reg_after, self.cat_after)]:
            reg_data, cat_data = data
            assert_array_equal(reg_data[reg_data.get_component(reg_data.extended_component_id).x], cat_data.id['x'])
            assert_array_equal(reg_data[reg_data.get_component(reg_data.extended_component_id).y], cat_data.id['y'])
