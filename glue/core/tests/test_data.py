# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103,R0903,R0904

from __future__ import absolute_import, division, print_function

import pytest
import numpy as np
from numpy.testing import assert_equal
from mock import MagicMock

from glue.external import six
from glue import core

from ..component import Component, DerivedComponent, CategoricalComponent, DateTimeComponent
from ..component_id import ComponentID
from ..component_link import ComponentLink, CoordinateComponentLink, BinaryComponentLink
from ..coordinates import Coordinates
from ..data import Data, pixel_label
from ..link_helpers import LinkSame
from ..data_collection import DataCollection
from ..exceptions import IncompatibleAttribute
from ..hub import Hub
from ..registry import Registry
from ..subset import (Subset, CategoricalROISubsetState, SubsetState,
                      RoiSubsetState, RangeSubsetState,
                      CategoricalMultiRangeSubsetState,
                      CategoricalROISubsetState2D, AndState)
from ..roi import PolygonalROI, CategoricalROI, RangeROI, RectangularROI

from .test_state import clone


class _TestCoordinates(Coordinates):

    def pixel2world(self, *args):
        return [(i + 2.) * a for i, a in enumerate(args)]

    def world2pixel(self, *args):
        return [a / (i + 2.) for i, a in enumerate(args)]


class TestData(object):

    def setup_method(self, method):
        self.data = Data(label="Test Data")
        Registry().clear()
        comp = Component(np.random.random((2, 3)))
        self.comp = comp
        self.data.coords = _TestCoordinates()
        self.comp_id = self.data.add_component(comp, 'Test Component')

    def test_2d_component_print(self):
        assert str(self.comp) == 'Component with shape (2, 3)'

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
        comp = Component(np.random.random((2, 3)))
        cid2 = self.data.add_component(comp, cid)
        assert cid2 is cid

    def test_add_component_via_setitem(self):
        d = Data(x=[1, 2, 3])
        d['y'] = d['x'] * 2
        np.testing.assert_array_equal(d['y'], [2, 4, 6])

    def test_add_component_incompatible_shape(self):
        comp = MagicMock()
        comp.data.shape = (3, 2)
        with pytest.raises(TypeError) as exc:
            self.data.add_component(comp("junk label"))
        if six.PY3:
            assert exc.value.args[0] == ("add_component() missing 1 required "
                                         "positional argument: 'label'")
        else:
            assert exc.value.args[0] == ("add_component() takes exactly 3 "
                                         "arguments (2 given)")

    def test_get_getitem_incompatible_attribute(self):
        cid = ComponentID('bad')
        with pytest.raises(IncompatibleAttribute) as exc:
            self.data.__getitem__(cid)
        assert exc.value.args[0] is cid

    def test_get_component_incompatible_attribute(self):
        cid = ComponentID('bad')
        with pytest.raises(IncompatibleAttribute) as exc:
            self.data.get_component(cid)
        assert exc.value.args[0] is cid

    def test_get_component_name(self):
        d = Data(x=[1, 2, 3])
        assert isinstance(d.get_component('x'), Component)

    def test_component_ids(self):
        cid = self.data.component_ids()
        assert self.comp_id in cid

    def test_new_subset(self):
        sub = self.data.new_subset()
        assert sub in self.data.subsets

    def test_data_not_created_with_subsets(self):
        assert len(self.data.subsets) == 0

    def test_register(self):
        hub = MagicMock(spec_set=Hub)
        self.data.register_to_hub(hub)
        assert hub is self.data.hub

    def test_component_order(self):
        """Components should be returned in the order they were specified"""
        data = Data()
        comp = Component(np.array([1, 2, 3]))
        labels = 'asldfkjaAREGWoibasiwnsldkgajsldkgslkg'
        for label in labels:
            data.add_component(comp, label)
        ids = data.visible_components
        assert [cid.label for cid in ids] == list(labels)

    def test_broadcast(self):
        hub = MagicMock(spec_set=Hub)

        # make sure broadcasting with no hub is ok
        self.data.broadcast('testing')

        # make sure broadcast with hub gets relayed
        self.data.register_to_hub(hub)
        self.data.broadcast('testing')
        assert hub.broadcast.call_count == 1

    def test_double_hub_add(self):
        hub = MagicMock(spec_set=Hub)
        hub2 = MagicMock(spec_set=Hub)
        self.data.register_to_hub(hub)
        with pytest.raises(AttributeError) as exc:
            self.data.__setattr__('hub', hub2)
        assert exc.value.args[0] == ("Data has already been assigned "
                                     "to a different hub")

    def test_primary_components(self):
        compid = ComponentID('virtual')
        link = MagicMock(spec_set=ComponentLink)
        comp = DerivedComponent(self.data, link)

        self.data.add_component(comp, compid)

        pricomps = self.data.primary_components
        print(self.comp_id, compid, pricomps)
        print(self.comp_id in pricomps)
        print(compid not in pricomps)
        assert self.comp_id in pricomps
        assert compid not in pricomps

    def test_add_component_invalid_component(self):
        comp = Component(np.array([1]))
        with pytest.raises(ValueError) as exc:
            self.data.add_component(comp, label='bad')
        assert exc.value.args[0].startswith("The dimensions of component bad")

    def test_add_component_link(self):
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
        comp.data.shape = self.data.shape
        self.data.add_component(comp, compid)

        result = self.data[compid]
        link.compute.assert_called_with(self.data)

    def test_find_component_id(self):
        cid = self.data.find_component_id('Test Component')
        assert cid == self.comp_id
        assert self.data.find_component_id('does not exist') is None

    def test_add_subset(self):
        s = Subset(Data())
        self.data.add_subset(s)
        assert s in self.data.subsets

    def test_add_subset_with_subset_state(self):
        """Passing a subset state auto-wraps into a subset object"""
        state = SubsetState()
        self.data.add_subset(state)
        added = self.data.subsets[-1]
        assert added.subset_state is state
        assert added.data is self.data

    def test_add_subset_reparents_subset(self):
        """add_subset method updates subset.data reference"""
        s = Subset(None)
        self.data.add_subset(s)
        assert s.data is self.data

    def test_add_subset_disambiguates_label(self):
        """adding subset should disambiguate label if needed"""
        s1 = Subset(None)
        self.data.add_subset(s1)
        s1.label = "test_subset_label"
        s2 = Subset(None)
        s2.label = "test_subset_label"
        assert s2.label == "test_subset_label"
        self.data.add_subset(s2)
        assert s2.label != "test_subset_label"

    def test_add_subset_with_hub(self):
        s = Subset(None)
        hub = MagicMock(spec_set=Hub)
        self.data.register_to_hub(hub)

        self.data.add_subset(s)
        assert s in self.data.subsets
        assert hub.broadcast.call_count == 1

    def test_remove_component(self):
        hub = MagicMock(spec_set=Hub)
        self.data.register_to_hub(hub)
        self.data.remove_component(self.comp_id)
        assert self.comp_id not in self.data.components
        assert hub.broadcast.call_count == 2

    def test_get_component(self):
        assert self.data.get_component(self.comp_id) is self.comp

    def test_get_None_component(self):

        with pytest.raises(IncompatibleAttribute):
            self.data.get_component(None)

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

    def test_fancy_view(self):
        result = self.data[self.comp_id, :, 2]
        np.testing.assert_array_equal(result, self.data[self.comp_id][:, 2])

    def test_get_by_string(self):
        result = self.data['Test Component']
        assert result is self.comp.data

    def test_get_by_missing_string(self):
        with pytest.raises(IncompatibleAttribute) as exc:
            result = self.data['xyz']
        assert exc.value.args[0] == 'xyz'

    def test_immutable(self):
        d = Data(x=[1, 2, 3])
        with pytest.raises(ValueError) as exc:
            d['x'][:] = 5
        assert 'read-only' in exc.value.args[0]
        assert not d['x'].flags['WRITEABLE']

    def test_categorical_immutable(self):
        d = Data()
        c = CategoricalComponent(['M', 'M', 'F'], categories=['M', 'F'])
        d.add_component(c, label='gender')

        with pytest.raises(ValueError) as exc:
            d['gender'][:] = 5
        assert 'read-only' in exc.value.args[0]
        assert not d['gender'].flags['WRITEABLE']

    def test_update_clears_subset_cache(self):
        from ..roi import RectangularROI

        d = Data(x=[1, 2, 3], y=[1, 2, 3])
        s = d.new_subset()
        state = core.subset.RoiSubsetState()
        state.xatt = d.id['x']
        state.yatt = d.id['y']
        state.roi = RectangularROI(xmin=1.5, xmax=2.5, ymin=1.5, ymax=2.5)
        s.subset_state = state

        np.testing.assert_array_equal(s.to_mask(), [False, True, False])
        d.update_components({d.id['x']: [10, 20, 30]})
        np.testing.assert_array_equal(s.to_mask(), [False, False, False])

    def test_add_derived_implicit(self):

        # Regression test for a bug that caused derived components added via
        # the data[...] = ... syntax to have links that did not include a 'to'
        # argument, leading the link manager to add a ghost component to the
        # data.

        from ..data_collection import DataCollection

        dc = DataCollection([])

        data = Data(x=[1, 2, 3], y=[2, 3, 4], label='data1')
        dc.append(data)

        data['z'] = data.id['x'] + 1

        # There should be five components: x, y, z, pixel, and world
        assert len(data.components) == 5

    def test_remove_derived_dependency(self):

        # Regression test for a bug that occurred when removing a component
        # used in a derived component, which should also remove the derived
        # component itself. To make things more fun, we set up a chain of
        # derived components to make sure they are all removed.

        data = Data(a=[1, 2, 3], b=[2, 3, 4], label='data1')

        data['c'] = data.id['a'] + 1
        data['d'] = data.id['c'] + 1
        data['e'] = data.id['d'] + 1
        data['f'] = data.id['e'] + 1

        a_id = data.id['a']
        b_id = data.id['b']
        c_id = data.id['c']
        d_id = data.id['d']
        e_id = data.id['e']
        f_id = data.id['f']

        # There should be five components: pixel, world, a, b, c, d, e, f
        assert len(data.components) == 8

        data.remove_component(data.id['d'])

        # This should also remove e and f since they depend on d

        assert len(data.components) == 5

        assert a_id in data.components
        assert b_id in data.components
        assert c_id in data.components
        assert d_id not in data.components
        assert e_id not in data.components
        assert f_id not in data.components

    def test_links_property(self):

        data = Data(a=[1, 2, 3], b=[2, 3, 4], label='data1')

        assert len(data.links) == 2
        assert isinstance(data.links[0], CoordinateComponentLink)
        assert isinstance(data.links[1], CoordinateComponentLink)

        data['c'] = data.id['a'] + 1

        assert len(data.links) == 3

        assert isinstance(data.links[2], BinaryComponentLink)


class TestROICreation(object):

    def test_range_roi(self):

        d = Data(xdata=[1, 2, 3], ydata=[1, 2, 3])
        comp = d.get_component(d.id['xdata'])
        roi = RangeROI('x', min=2, max=3)
        s = comp.subset_from_roi('xdata', roi)
        assert isinstance(s, RangeSubsetState)
        np.testing.assert_array_equal((s.lo, s.hi),
                                      [2, 3])

        roi = RangeROI('y', min=2, max=3)
        s = comp.subset_from_roi('xdata', roi, other_att='ydata',
                                 other_comp=d.get_component(d.id['ydata']))
        assert isinstance(s, RangeSubsetState)
        assert s.att == 'ydata'

    def test_range_roi_categorical(self):

        d = Data(x=['a', 'b', 'c'], y=[1, 2, 3])
        comp = d.get_component(d.id['x'])
        roi = CategoricalROI(['b', 'c'])
        s = comp.subset_from_roi(d.id['x'], roi)
        assert isinstance(s, CategoricalROISubsetState)
        np.testing.assert_array_equal((s.roi.contains(['a', 'b', 'c'], None)),
                                      [False, True, True])

        roi = RangeROI('x', min=1, max=3)
        s = comp.subset_from_roi('x', roi)
        assert isinstance(s, CategoricalROISubsetState)
        np.testing.assert_array_equal((s.roi.contains(['a', 'b', 'c'], None)),
                                      [False, True, True])

    def test_polygon_roi(self):

        d = Data(x=[1, 1.3, 3, 10], y=[1, 1.5, 3, 10])
        x_comp = d.get_component(d.id['x'])
        y_comp = d.get_component(d.id['y'])
        roi = PolygonalROI([0, 0, 2, 2], [0, 2, 2, 0])
        s = x_comp.subset_from_roi(d.id['x'], roi, other_comp=y_comp, other_att=d.id['y'])
        assert isinstance(s, RoiSubsetState)
        np.testing.assert_array_equal(s.to_mask(d), [True, True, False, False])

    def test_polygon_categorical_rectangular(self):

        d = Data(x=[1, 1.3, 3, 10], y=['a', 'b', 'c', 'd'])
        x_comp = d.get_component(d.id['x'])
        y_comp = d.get_component(d.id['y'])
        roi = PolygonalROI([0, 0, 2, 2], [0, 2, 2, 0])
        s = x_comp.subset_from_roi('x', roi, other_comp=y_comp, other_att='y')
        assert isinstance(s, CategoricalMultiRangeSubsetState)

        np.testing.assert_array_equal(s.to_mask(d), [True, True, False, False])

    def test_polygon_categorical_arbitrary(self):

        d = Data(x=[1, 1.3, 3, 10], y=['a', 'b', 'c', 'd'])
        x_comp = d.get_component(d.id['x'])
        y_comp = d.get_component(d.id['y'])
        roi = PolygonalROI([0, 4, 4, 1, 0], [-0.5, 3.5, 0, -1, -0.5])
        s = x_comp.subset_from_roi('x', roi, other_comp=y_comp, other_att='y')
        assert isinstance(s, CategoricalMultiRangeSubsetState)

        np.testing.assert_array_equal(s.to_mask(d), [True, False, True, False])

    def test_rectangular_categorical(self):

        d = Data(x=[1, 1.3, 3, 10], y=['a', 'b', 'c', 'd'])
        x_comp = d.get_component(d.id['x'])
        y_comp = d.get_component(d.id['y'])
        roi = RectangularROI(xmin=-0.1, xmax=2.1, ymin=-0.1, ymax=2.1)
        s = x_comp.subset_from_roi('x', roi, other_comp=y_comp, other_att='y')
        assert isinstance(s, AndState)

        np.testing.assert_array_equal(s.to_mask(d), [True, True, False, False])

        s = y_comp.subset_from_roi('y', roi, other_comp=x_comp, other_att='x')
        assert isinstance(s, AndState)

        np.testing.assert_array_equal(s.to_mask(d), [True, True, False, False])

    def test_polygon_both_categorical_arbitrary(self):

        d = Data(x=['a', 'b', 'c', 'd', 'b', 'c'], y=['p', 'q', 'r', 's', 's', 'q'])
        x_comp = d.get_component(d.id['x'])
        y_comp = d.get_component(d.id['y'])
        roi = PolygonalROI([0.5, 1.5, 2.5, 1, 0.5], [0.5, 0.5, 2.5, 3.5, 0.5])
        s = x_comp.subset_from_roi('x', roi, other_comp=y_comp, other_att='y')
        assert isinstance(s, CategoricalROISubsetState2D)

        np.testing.assert_array_equal(s.to_mask(d), [False, True, True, False, True, False])

    def test_polygon_both_categorical_empty(self):

        d = Data(x=['a', 'b', 'c', 'd', 'b', 'c'], y=['p', 'q', 'r', 's', 's', 'q'])
        x_comp = d.get_component(d.id['x'])
        y_comp = d.get_component(d.id['y'])
        roi = PolygonalROI([0.5, 0.6, 0.6, 0.5], [0.5, 0.5, 0.6, 0.5])
        s = x_comp.subset_from_roi('x', roi, other_comp=y_comp, other_att='y')
        assert isinstance(s, CategoricalROISubsetState2D)

        np.testing.assert_array_equal(s.to_mask(d), [False, False, False, False, False, False])


def test_component_id_item_access():

    data = Data()

    c1 = Component(np.array([1, 2, 3]))
    data.add_component(c1, 'values')

    c2 = Component(np.array([4., 5., 6.]))
    data.add_component(c2, 'Flux')

    assert data.id['values'] == data.find_component_id('values')
    assert data.id['Flux'] == data.find_component_id('Flux')


def test_component_id_item_access_missing():
    """id attribute should raise KeyError if requesting a bad ComponentID"""
    data = Data()
    with pytest.raises(KeyError):
        data.id['not found']


class TestPixelLabel(object):

    def test(self):
        assert pixel_label(0, 2) == "0 [y]"
        assert pixel_label(1, 2) == "1 [x]"
        assert pixel_label(0, 3) == "0 [z]"
        assert pixel_label(1, 3) == "1 [y]"
        assert pixel_label(2, 3) == "2 [x]"
        assert pixel_label(1, 0) == "1"
        assert pixel_label(1, 4) == "1"


@pytest.mark.parametrize(('kwargs'),
                         [{'x': [1, 2, 3]},
                          {'x': np.array([1, 2, 3])},
                          {'x': [[1, 2, 3], [2, 3, 4]]},
                          {'x': [1, 2], 'y': [2, 3]}])
def test_init_with_inputs(kwargs):
    """Passing array-like objects as keywords to Data
    auto-populates Components with label names = keywords"""
    d = Data(**kwargs)
    for label, data in kwargs.items():
        np.testing.assert_array_equal(d[d.id[label]], data)


def test_init_with_invalid_kwargs():
    with pytest.raises(ValueError) as exc:
        d = Data(x=[1, 2], y=[1, 2, 3])
    assert exc.value.args[0].startswith('The dimensions of component')


def test_getitem_with_component_link():
    d = Data(x=[1, 2, 3, 4])
    y = d.id['x'] * 5
    np.testing.assert_array_equal(d[y], [5, 10, 15, 20])


def test_getitem_with_component_link_and_slice():
    d = Data(x=[1, 2, 3, 4])
    y = d.id['x'] * 5
    np.testing.assert_array_equal(d[y, ::2], [5, 15])


def test_add_link_with_binary_link():
    d = Data(x=[1, 2, 3, 4], y=[4, 5, 6, 7])
    z = d.id['x'] + d.id['y']
    d.add_component_link(z, 'z')
    np.testing.assert_array_equal(d[d.id['z']], [5, 7, 9, 11])


def test_foreign_pixel_components_not_in_visible():
    """Pixel components from other data should not be visible"""

    # currently, this is trivially satisfied since all coordinates are hidden

    d1 = Data(x=[1], y=[2])
    d2 = Data(w=[3], v=[4])
    dc = DataCollection([d1, d2])
    dc.add_link(LinkSame(d1.id['x'], d2.id['w']))

    dc.add_link(LinkSame(d1.get_world_component_id(0),
                         d2.get_world_component_id(0)))

    assert d2.get_pixel_component_id(0) not in d1.visible_components
    np.testing.assert_array_equal(d1[d2.get_pixel_component_id(0)], [0])


def test_add_binary_component():

    d = Data(x=[1, 2, 3], y=[2, 3, 4])
    z = d.id['x'] + d.id['y']
    d.add_component(z, label='z')

    np.testing.assert_array_equal(d['z'], [3, 5, 7])


EXPECTED_STR = """
Data Set: mydata
Number of dimensions: 1
Shape: 3
Main components:
 - x
 - y
Coordinate components:
 - Pixel Axis 0 [x]
 - World 0
""".strip()


def test_data_str():
    # Regression test for Data.__str__
    d = Data(x=[1, 2, 3], y=[2, 3, 4], label='mydata')
    assert str(d) == EXPECTED_STR


EXPECTED_STR_WITH_DERIVED = """
Data Set: mydata
Number of dimensions: 1
Shape: 3
Main components:
 - x
 - y
Derived components:
 - z
Coordinate components:
 - Pixel Axis 0 [x]
 - World 0
""".strip()


def test_data_str_with_derived():
    d = Data(x=[1, 2, 3], y=[2, 3, 4], label='mydata')
    d['z'] = d.id['x'] + 1
    assert str(d) == EXPECTED_STR_WITH_DERIVED


def test_update_values_from_data():
    d1 = Data(a=[1, 2, 3], b=[4, 5, 6], label='banana')
    d2 = Data(b=[1, 2, 3, 4], c=[5, 6, 7, 8], label='apple')
    d1a = d1.id['a']
    d1b = d1.id['b']
    d2b = d2.id['b']
    d2c = d2.id['c']
    d1.update_values_from_data(d2)
    assert d1a not in d1.components
    assert d1b in d1.components
    assert d2b not in d1.components
    assert d2c not in d1.components
    assert [cid.label for cid in d1.visible_components] == ['b', 'c']
    assert d1.shape == (4,)


def test_update_values_from_data_invalid():

    d1 = Data(a=[1, 2, 3], label='banana')
    d1.add_component([3, 4, 5], 'a')
    d2 = Data(b=[1, 2, 3, 4], c=[5, 6, 7, 8], label='apple')
    with pytest.raises(ValueError) as exc:
        d1.update_values_from_data(d2)
    assert exc.value.args[0] == "Non-unique component labels in original data"

    d1 = Data(a=[1, 2, 3], b=[4, 5, 6], label='banana')
    d2 = Data(b=[1, 2, 3, 4], label='apple')
    d2.add_component([5, 6, 7, 8], 'b')
    with pytest.raises(ValueError) as exc:
        d1.update_values_from_data(d2)
    assert exc.value.args[0] == "Non-unique component labels in new data"


def test_update_values_from_data_order():

    # Make sure that the order of components is preserved when calling
    # Data.update_values_from_data. The final order should be first
    # components that existed before, followed by new components

    d1 = Data()
    d1['c'] = [1, 2, 3]
    d1['b'] = [2, 3, 4]
    d1['j'] = [0, 1, 2]
    d1['a'] = [4, 4, 4]
    d1['f'] = [4, 5, 6]

    d2 = Data()
    d2['h'] = [4, 4, 4]
    d2['j'] = [0, 1, 2]
    d2['a'] = [4, 4, 4]

    d2.update_values_from_data(d1)

    assert [cid.label for cid in d2.visible_components] == ['j', 'a', 'c', 'b', 'f']


def test_find_component_id_with_cid():

    # Regression test for a bug that caused Data.find_component_id to return
    # True erroneously when passing a component ID.

    d1 = Data()
    d1['a'] = ['a', 'b', 'c']
    d1['b'] = [1, 2, 3]

    assert d1.find_component_id(d1.id['a']) is d1.id['a']
    assert d1.find_component_id(d1.id['b']) is d1.id['b']


def test_parent_preserved_session():

    # Regression test for a bug that caused ComponentID parents to not be
    # preserved when saving and restoring a session.

    from ..link_helpers import LinkSame
    from ..data_collection import DataCollection

    d1 = Data(x=[1], y=[2], label='test1')
    d2 = Data(w=[3], v=[4], label='test2')

    dc = DataCollection([d1, d2])
    dc.add_link(LinkSame(d1.id['x'], d2.id['w']))

    assert d1.id['x'].parent is d1
    assert d1.id['y'].parent is d1

    assert d2.id['w'].parent is d2
    assert d2.id['v'].parent is d2

    dc2 = clone(dc)

    assert dc2[0].id['x'].parent.label == 'test1'
    assert dc2[0].id['y'].parent.label == 'test1'

    assert dc2[1].id['w'].parent.label == 'test2'
    assert dc2[1].id['v'].parent.label == 'test2'


def test_preserve_datetime():
    # Make sure that we recognize and preserve the Numpy datetime64 format
    dates = np.array([1, 2, 3], dtype='M8[D]')
    data = Data(dates=dates)
    assert isinstance(data.get_component('dates'), DateTimeComponent)


def test_clone_meta():
    # Regression test for a bug that caused metadata to not be preserved
    # when saving/loading sessions.
    class CustomObject(object):
        pass
    data1 = Data(x=[1, 2, 3])
    data1.meta['a'] = 1
    data1.meta['b'] = 'test'
    data1.meta['c'] = CustomObject()
    data2 = clone(data1)
    assert data2.meta['a'] == 1
    assert data2.meta['b'] == 'test'
    assert 'c' not in data2.meta


def test_update_coords():

    # Make sure that when overriding coords, the world coordinate components
    # are updated.

    data1 = Data(x=[1, 2, 3])

    assert len(data1.components) == 3

    assert_equal(data1[data1.world_component_ids[0]], [0, 1, 2])

    data2 = Data(x=[1, 2, 3])

    assert len(data1.links) == 2
    assert len(data2.links) == 2

    data_collection = DataCollection([data1, data2])

    assert len(data_collection.links) == 4

    data_collection.add_link(LinkSame(data1.world_component_ids[0], data2.world_component_ids[0]))

    assert len(data_collection.links) == 5

    class CustomCoordinates(Coordinates):

        def axis_label(self, axis):
            return 'Custom {0}'.format(axis)

        def world2pixel(self, *world):
            return tuple([0.4 * w for w in world])

        def pixel2world(self, *pixel):
            return tuple([2.5 * p for p in pixel])

    data1.coords = CustomCoordinates()

    assert len(data1.components) == 3

    assert_equal(data1[data1.world_component_ids[0]], [0, 2.5, 5])

    assert sorted(cid.label for cid in data1.world_component_ids) == ['Custom 0']

    # The link between the two world coordinates should be remove
    assert len(data_collection.links) == 4
