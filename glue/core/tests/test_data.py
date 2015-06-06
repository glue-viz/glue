# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103,R0903,R0904

from __future__ import absolute_import, division, print_function

import pytest
import numpy as np
from mock import MagicMock

from ..data import (ComponentID, Component, Data,
                    DerivedComponent, pixel_label, CategoricalComponent)
from ..coordinates import Coordinates
from ..subset import Subset, SubsetState
from ..hub import Hub, HubListener
from ..exceptions import IncompatibleAttribute
from ..component_link import ComponentLink
from ..registry import Registry
from ... import core
from ...external import six


class TestCoordinates(Coordinates):

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
        self.data.coords = TestCoordinates()
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
        if isinstance(exc.value, six.string_types):  # python 2.6
            assert exc.value == ("add_component() takes at least 3 "
                                 "arguments (2 given)")
        elif six.PY3:
            assert exc.value.args[0] == ("add_component() missing 1 required "
                                         "positional argument: 'label'")
        else:
            assert exc.value.args[0] == ("add_component() takes at least 3 "
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
        self.data.remove_component(self.comp_id)
        assert not self.comp_id in self.data.components

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
        try:
            assert 'read-only' in exc.value.args[0]
        except AttributeError:  # COMPAT: Python 2.6
            assert 'read-only' in exc.value
        assert not d['x'].flags['WRITEABLE']

    def test_categorical_immutable(self):
        d = Data()
        c = CategoricalComponent(['M', 'M', 'F'], categories=['M', 'F'])
        d.add_component(c, label='gender')

        with pytest.raises(ValueError) as exc:
            d['gender'][:] = 5
        try:
            assert 'read-only' in exc.value.args[0]
        except AttributeError:  # COMPAT: Python 2.6
            assert 'read-only' in exc.value
        assert not d['gender'].flags['WRITEABLE']

    def test_update_clears_subset_cache(self):
        from glue.core.roi import RectangularROI

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
        assert pixel_label(0, 2) == "y"
        assert pixel_label(1, 2) == "x"
        assert pixel_label(0, 3) == "z"
        assert pixel_label(1, 3) == "y"
        assert pixel_label(2, 3) == "x"
        assert pixel_label(1, 0) == "Axis 1"
        assert pixel_label(1, 4) == "Axis 1"


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
    from ..link_helpers import LinkSame
    from ..data_collection import DataCollection

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
