import numpy as np
import json
import pytest

from ..state import GlueSerializer, GlueUnSerializer
from ... import core
from ...qt.glue_application import GlueApplication
from ...qt.widgets.scatter_widget import ScatterWidget
from ...qt.widgets.image_widget import ImageWidget
from ...qt.widgets.histogram_widget import HistogramWidget


def clone(object):
    gs = GlueSerializer()
    oid = gs.id(object)
    dump1 = gs.do_all()
    state = json.dumps(dump1, default=gs.json_default)
    gu = GlueUnSerializer(state)
    result = gu.object(oid)
    return result


def dump(object):
    gs = GlueSerializer()
    oid = gs.id(object)
    return gs.do_all()


def test_data():
    d = core.Data(x=[1, 2, 3], label='testing')
    d2 = clone(d)
    assert d2.label == 'testing'
    np.testing.assert_array_equal(d2['x'], [1, 2, 3])
    np.testing.assert_array_equal(d2['Pixel Axis 0'], [0, 1, 2])


def test_inequality_subset():
    d = core.Data(x=[1, 2, 3], label='testing')
    s = d.new_subset(label='abc')
    s.subset_state = d.id['x'] > 1
    d2 = clone(d)
    s2 = d2.subsets[0]
    assert s.label == s2.label
    np.testing.assert_array_equal(s2.to_mask(), [False, True, True])
    assert s.style == s2.style


def test_compound_state():
    d = core.Data(x=[1, 2, 3])
    s = d.new_subset(label='abc')
    s.subset_state = (d.id['x'] > 2) | (d.id['x'] < 1.5)
    d2 = clone(d)
    np.testing.assert_array_equal(d2.subsets[0].to_mask(), [True, False, True])


def test_empty_subset():
    d = core.Data(x=[1, 2, 3], label='testing')
    s = d.new_subset(label='abc')
    s.style.color = 'blue'
    s2 = clone(s)
    assert s.style == s2.style
    assert s2.style.color == 'blue'


def test_box_roi_subset():
    d = core.Data(x=[1, 2, 3], y=[1, 2, 3])
    s = d.new_subset(label='box')
    roi = core.roi.RectangularROI(xmin=1.1, xmax=2.1, ymin=1.1, ymax=2.1)
    s.subset_state = core.subset.RoiSubsetState(xatt=d.id['x'],
                                                yatt=d.id['y'], roi=roi)

    d2 = clone(d)
    np.testing.assert_array_equal(
        d2.subsets[0].to_mask(), [False, True, False])


def test_range_roi():
    roi = core.roi.RangeROI('x', min=1, max=2)
    r2 = clone(roi)
    assert r2.ori == 'x'
    assert r2.min == 1
    assert r2.max == 2


def test_circular_roi():
    roi = core.roi.CircularROI(xc=0, yc=1, radius=2)
    r2 = clone(roi)
    assert r2.xc == 0
    assert r2.yc == 1
    assert r2.radius == 2


def test_polygonal_roi():
    roi = core.roi.PolygonalROI()
    roi.add_point(0, 0)
    roi.add_point(0, 1)
    roi.add_point(1, 0)
    r2 = clone(roi)
    assert r2.vx == [0, 0, 1]
    assert r2.vy == [0, 1, 0]


class TestApplication(object):

    def check_clone(self, app):
        copy = clone(app)
        hub1 = app.session.hub
        hub2 = copy.session.hub
        assert len(hub1._subscriptions) == len(hub2._subscriptions)

        # data collections are the same
        for d1, d2 in zip(app.session.data_collection,
                          copy.session.data_collection):
            assert d1.label == d2.label
            for cid1, cid2 in zip(d1.components, d2.components):
                assert cid1.label == cid2.label
                np.testing.assert_array_equal(d1[cid1], d2[cid2])

        # same data viewers, in the same tabs
        for tab1, tab2 in zip(app.viewers, copy.viewers):
            assert len(tab1) == len(tab2)
            for v1, v2 in zip(tab1, tab2):
                assert type(v1) == type(v2)
                # same window properties
                assert v1.viewer_size == v2.viewer_size
                assert v1.position == v2.position

                # same viewer-level properties (axis label, scaling, etc)
                assert set(v1.properties.keys()) == set(v2.properties.keys())
                for k in v1.properties:
                    if hasattr(v1.properties[k], 'label'):
                        assert v1.properties[k].label == v2.properties[k].label
                    else:
                        assert v1.properties[k] == v2.properties[k]

                assert len(v1.layers) == len(v2.layers)
                for l1, l2 in zip(v1.layers, v2.layers):
                    assert l1.layer.label == l2.layer.label  # same data/subset
                    assert l1.layer.style == l2.layer.style

        return copy

    def test_bare_application(self):
        app = GlueApplication()
        self.check_clone(app)

    def test_data_application(self):
        dc = core.DataCollection([core.Data(label='test',
                                            x=[1, 2, 3], y=[2, 3, 4])])
        app = GlueApplication(dc)
        self.check_clone(app)

    def test_links(self):
        d1 = core.Data(label='x', x=[1, 2, 3])
        d2 = core.Data(label='y', y=[3, 4, 8])
        dc = core.DataCollection([d1, d2])
        link = core.ComponentLink([d1.id['x']], d2.id['y'], lambda x: 2 * x)
        dc.add_link(link)

        np.testing.assert_array_equal(d1['y'], [2, 4, 6])

        app = GlueApplication(dc)
        self.check_clone(app)

    def test_scatter_viewer(self):
        d = core.Data(label='x', x=[1, 2, 3, 4, 5], y=[2, 3, 4, 5, 6])
        dc = core.DataCollection([d])
        app = GlueApplication(dc)
        w = app.new_data_viewer(ScatterWidget, data=d)
        self.check_clone(app)

        s1 = d.new_subset(label='testing 123')
        s2 = d.new_subset(label='testing 234')
        assert len(w.layers) == 3
        l1, l2, l3 = w.layers
        l1.zorder, l2.zorder = l2.zorder, l1.zorder
        l3.visible = False
        assert l3.visible is False
        copy = self.check_clone(app)
        assert copy.viewers[0][0].layers[-1].visible is False

    def test_image_viewer(self):
        LinkSame = core.link_helpers.LinkSame

        d = core.Data(label='im', x=[[1, 2], [2, 3]])
        d2 = core.Data(label='cat',
                       x=[0, 1, 0, 1],
                       y=[0, 0, 1, 1],
                       z=[1, 2, 3, 4])

        dc = core.DataCollection([d, d2])
        dc.add_link(LinkSame(d.get_pixel_component_id(0), d2.id['x']))
        dc.add_link(LinkSame(d.get_pixel_component_id(1), d2.id['y']))

        app = GlueApplication(dc)
        w = app.new_data_viewer(ImageWidget, data=d)
        self.check_clone(app)

        s = d.new_subset(label='testing')
        assert len(w.layers) == 2
        self.check_clone(app)

        # add scatter layer
        s.label = 'testing 2'
        l = w.add_layer(d2)
        assert len(w.layers) == 3
        self.check_clone(app)

        # add RGB Image
        s.label = 'testing 3'
        x = d.id['x']
        l = w.client.add_rgb_layer(d, r=x, g=x, b=x)
        l.layer_visible['blue'] = False

        assert l is not None
        clone = self.check_clone(app)
        assert clone.viewers[0][0].layers[-1].layer_visible['blue'] is False

    def test_histogram(self):
        d = core.Data(label='hist', x=[[1, 2], [2, 3]])
        dc = core.DataCollection([d])

        app = GlueApplication(dc)
        w = app.new_data_viewer(HistogramWidget, data=d)
        self.check_clone(app)

        s = d.new_subset(label='wxy')
        assert len(w.layers) == 2
        self.check_clone(app)
