import pytest
import numpy as np
from mock import MagicMock

from ..data import ComponentID, Component, Data, DerivedComponent, pixel_label
from ..coordinates import Coordinates
from ..subset import Subset
from ..hub import Hub
from ..exceptions import IncompatibleAttribute
from ..component_link import ComponentLink
from ..coordinates import WCSCoordinates, WCSCubeCoordinates
from ...tests import example_data
from ..registry import Registry

class TestCoordinates(Coordinates):

    def pixel2world(self, *args):
        return [(i + 2.) * a for i, a in enumerate(args)]

    def world2pixel(self, *args):
        return [a / (i + 2.) for i, a in enumerate(args)]


class TestData(object):

    def setup_method(self, method):
        self.data = Data(label="Test Data")
        Registry().clear()
        comp = MagicMock()
        comp.data.shape = (2, 3)
        comp.shape = (2, 3)
        comp.units = None
        self.comp = comp
        self.data.coords = TestCoordinates()
        self.comp_id = self.data.add_component(comp, 'Test Component')

    def test_shape_empty(self):
        d = Data()
        assert d.shape == ()

    def test_ndim_empty(self):
        d = Data()
        assert d.ndim == 0

    def test_shape(self):
        assert self.data.shape == (2, 3)

    def test_ndim(self):
        assert self.data.ndim == 2

    def test_size(self):
        assert self.data.size == 6

    def test_label(self):
        d = Data()
        assert d.label == ''
        assert self.data.label == "Test Data"

    def test_set_label(self):
        d = Data()
        d.label = 'test_set_label'
        assert d.label == 'test_set_label'

    def test_add_component_with_id(self):
        cid = ComponentID("test")
        comp = MagicMock()
        comp.shape = (2, 3)
        comp.units = None
        cid2 = self.data.add_component(comp, cid)
        assert cid2 is cid

    def test_add_component_incompatible_shape(self):
        comp = MagicMock()
        comp.data.shape = (3, 2)
        with pytest.raises(TypeError) as exc:
            self.data.add_component(comp("junk label"))
        assert exc.value.args[0] == "add_component() takes exactly 3 arguments (2 given)"

    def test_get_getitem_incompatible_attribute(self):
        cid = ComponentID('bad')
        with pytest.raises(IncompatibleAttribute) as exc:
            self.data.__getitem__(cid)
        assert exc.value.args[0] == "bad not in data set Test Data"

    def test_get_component_incompatible_attribute(self):
        cid = ComponentID('bad')
        with pytest.raises(IncompatibleAttribute) as exc:
            self.data.get_component(cid)
        assert exc.value.args[0] == "bad not in data set"

    def test_component_ids(self):
        cid = self.data.component_ids()
        assert self.comp_id in cid

    def test_new_subset(self):
        sub = self.data.new_subset()
        assert sub in self.data.subsets

    def test_data_created_with_edit_subset(self):
        assert len(self.data.subsets) == 1

    def test_register(self):
        hub = MagicMock(spec_set=Hub)
        not_hub = MagicMock()
        self.data.register_to_hub(hub)
        assert hub is self.data.hub
        with pytest.raises(TypeError) as exc:
            self.data.register_to_hub(not_hub)
        assert exc.value.args[0].startswith("input is not a Hub object")

    def test_component_order(self):
        """Components should be returned in alphabetical order"""
        data = Data()
        comp = Component(np.array([1, 2, 3]))
        labels = 'asldfkjaAREGWoibasiwnsldkgajsldkgslkg'
        for label in labels:
            data.add_component(comp, label)
        ids = data.components
        labels = [cid.label.lower() for cid in ids]
        assert labels == sorted(labels)

    def test_broadcast(self):
        hub = MagicMock(spec_set=Hub)

        # make sure broadcasting with no hub is ok
        self.data.broadcast()

        # make sure broadcast with hub gets relayed
        self.data.register_to_hub(hub)
        self.data.broadcast()
        assert hub.broadcast.call_count == 1

    def test_clone_subset(self):
        sub1 = self.data.new_subset()
        sub2 = self.data.create_subset_from_clone(sub1)
        assert sub1.subset_state is sub2.subset_state

    def test_double_hub_add(self):
        hub = MagicMock(spec_set=Hub)
        hub2 = MagicMock(spec_set=Hub)
        self.data.register_to_hub(hub)
        with pytest.raises(AttributeError) as exc:
            self.data.__setattr__('hub', hub2)
        assert exc.value.args[0] == "Data has already been assigned to a different hub"

    def test_primary_components(self):
        compid = ComponentID('virtual')
        link = MagicMock(spec_set=ComponentLink)
        comp = DerivedComponent(self.data, link)

        self.data.add_component(comp, compid)

        pricomps = self.data.primary_components
        print self.comp_id, compid, pricomps
        print self.comp_id in pricomps
        print compid not in pricomps
        assert self.comp_id in pricomps
        assert compid not in pricomps

    def test_add_component_invalid_label(self):
        with pytest.raises(TypeError) as exc:
            self.data.add_component(self.comp, label=5)
        assert exc.value.args[0] == "label must be a ComponentID or string"

    def test_add_component_invalid_component(self):
        comp = Component(np.array([1]))
        with pytest.raises(TypeError) as exc:
            self.data.add_component(comp, label='bad')
        assert exc.value.args[0] == "Compoment is incompatible with other components in this data"

    def test_add_component_link(self):
        compid = ComponentID('virtual')
        link = MagicMock(spec_set=ComponentLink)
        cid = ComponentID("new id")
        link.get_to_id.return_value = cid

        self.data.add_component_link(link)
        assert cid in self.data.derived_components

    def test_derived_components(self):
        compid = ComponentID('virtual')
        link = MagicMock(spec_set=ComponentLink)
        comp = DerivedComponent(self.data, link)

        self.data.add_component(comp, compid)

        pricomps = self.data.derived_components
        assert self.comp_id not in pricomps
        assert compid in pricomps

    def test_str_empty(self):
        d = Data()
        str(d)

    def test_str_(self):
        str(self.data)

    def test_add_derived_component(self):
        compid = ComponentID('virtual')
        link = MagicMock(spec_set=ComponentLink)
        comp = DerivedComponent(self.data, link)
        self.data.add_component(comp, compid)

        result = self.data[compid]
        link.compute.assert_called_once_with(self.data)

    def test_find_component_id(self):
        cid = self.data.find_component_id('Test Component')
        assert cid == [self.comp_id]

    def test_add_subset(self):
        s = MagicMock(spec_set=Subset)
        self.data.add_subset(s)
        assert s in self.data.subsets

    def test_add_subset_with_hub(self):
        s = MagicMock(spec_set=Subset)
        hub = MagicMock(spec_set=Hub)
        self.data.register_to_hub(hub)

        self.data.add_subset(s)
        assert s in self.data.subsets
        assert hub.broadcast.call_count == 1

    def test_remove_component(self):
        self.data.remove_component(self.comp_id)
        assert not self.comp_id in self.data.components

    def test_get_component(self):
        assert self.data.get_component(self.comp_id) is self.comp

    def test_get_item(self):
        assert self.data[self.comp_id] is self.comp.data

    def test_coordinate_links(self):
        links = self.data.coordinate_links
        w0 = self.data[self.data.get_world_component_id(0)]
        w1 = self.data[self.data.get_world_component_id(1)]
        p0 = self.data[self.data.get_pixel_component_id(0)]
        p1 = self.data[self.data.get_pixel_component_id(1)]

        w0prime = links[0].compute(self.data)
        p0prime = links[1].compute(self.data)
        w1prime = links[2].compute(self.data)
        p1prime = links[3].compute(self.data)

        np.testing.assert_array_equal(w0, w0prime)
        np.testing.assert_array_equal(w1, w1prime)
        np.testing.assert_array_equal(p0, p0prime)
        np.testing.assert_array_equal(p1, p1prime)

    def test_coordinate_links_empty_data(self):
        d = Data()
        d.coords = None
        assert d.coordinate_links == []

    def test_coordinate_links_idempotent(self):
        """Should only calculate links once, and
        return the same objects every time"""
        links = self.data.coordinate_links
        links2 = self.data.coordinate_links
        assert links == links2

#XXX need to get data methods to work
#class TestGriddedData(object):
#    def test_parse_coords_2d(self):
#        """Valid fits header should parse into WCSCoordinates"""
#        data = example_data.simple_image()
#        assert isinstance(self.data.coords, WCSCoordinates)
#
#    def test_parse_coords_3d(self):
#        """Valid fits header should parse into WCSCoordinates"""
#        data = example_data.simple_cube()
#        assert isinstance(self.data.coords, WCSCubeCoordinates)


class TestPixelLabel(object):

    def test(self):
        assert pixel_label(0, 2) == "y"
        assert pixel_label(1, 2) == "x"
        assert pixel_label(0, 3) == "z"
        assert pixel_label(1, 3) == "y"
        assert pixel_label(2, 3) == "x"
        assert pixel_label(1, 0) == "Axis 1"
        assert pixel_label(1, 4) == "Axis 1"
