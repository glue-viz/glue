import numpy as np
import json
import pytest

from ..state import (GlueSerializer, GlueUnSerializer,
                     saver, loader, VersionedDict)

from ... import core
from ...qt.glue_application import GlueApplication
from ...qt.widgets.scatter_widget import ScatterWidget
from ...qt.widgets.image_widget import ImageWidget
from ...qt.widgets.histogram_widget import HistogramWidget
from .util import make_file
from ..data_factories import load_data
from .test_data_factories import TEST_FITS_DATA


def clone(object):
    gs = GlueSerializer(object)
    oid = gs.id(object)
    dump = gs.dumps()
    gu = GlueUnSerializer.loads(dump)
    result = gu.object(oid)
    return result


def test_none():
    assert clone(None) is None


def test_data():
    d = core.Data(x=[1, 2, 3], label='testing')
    d2 = clone(d)
    assert d2.label == 'testing'

    np.testing.assert_array_equal(d2['x'], [1, 2, 3])
    np.testing.assert_array_equal(d2['Pixel Axis 0'], [0, 1, 2])


def test_data_factory():
    with make_file(TEST_FITS_DATA, '.fits') as infile:
        d = load_data(infile)
        d2 = clone(d)

    np.testing.assert_array_equal(d['PRIMARY'], d2['PRIMARY'])


def test_save_numpy_scalar():
    assert clone(np.float32(5)) == 5


def tests_data_factory_double():

    from cStringIO import StringIO
    from astropy.io import fits
    d = np.random.normal(0, 1, (100, 100, 100))
    s = StringIO()
    fits.writeto(s, d)

    with make_file(s.getvalue(), '.fits') as infile:
        d = load_data(infile)
        d2 = clone(d)

    assert len(GlueSerializer(d).dumps()) < \
        1.1 * len(GlueSerializer(d2).dumps())


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
    d = core.Data(x=[1, 2, 3], y=[2, 4, 8])
    s = d.new_subset(label='box')
    roi = core.roi.RectangularROI(xmin=1.1, xmax=2.1, ymin=2.2, ymax=4.2)
    s.subset_state = core.subset.RoiSubsetState(xatt=d.id['x'],
                                                yatt=d.id['y'], roi=roi)

    d2 = clone(d)
    np.testing.assert_array_equal(
        d2.subsets[0].to_mask(), [False, True, False])


def test_complex_state():
    d = core.Data(x=[1, 2, 3], y=[2, 4, 8])
    s = d.new_subset(label='test')
    s.subset_state = (d.id['x'] > 2) | (d.id['y'] < 4)
    s.subset_state = s.subset_state & (d.id['x'] < 4)

    d2 = clone(d)
    s2 = d2.subsets[0]
    np.testing.assert_array_equal(s2.to_mask(), [True, False, True])


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
        w.client.display_data = d
        l = w.client.rgb_mode(True)
        l.r = x
        l.g = x
        l.b = x
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

        w.nbins = 7
        self.check_clone(app)


class TestVersioning(object):

    def setup_method(self, method):

        @saver(core.Data, version=2)
        def s(d, context):
            return dict(v=2)

        @loader(core.Data, version=2)
        def l(d, context):
            return 2

        @saver(core.Data, version=3)
        def s(d, context):
            return dict(v=3)

        @loader(core.Data, version=3)
        def l(rec, context):
            return 3

    def teardown_method(self, method):
        GlueSerializer.dispatch._data[core.Data].pop(2)
        GlueSerializer.dispatch._data[core.Data].pop(3)
        GlueUnSerializer.dispatch._data[core.Data].pop(2)
        GlueUnSerializer.dispatch._data[core.Data].pop(3)

    def test_defualt_latest_save(self):
        assert GlueSerializer(core.Data()).dumpo().values()[0]['v'] == 3

    def test_legacy_load(self):
        data = json.dumps({'': {'_type': 'glue.core.Data',
                                '_protocol': 2, 'v': 2}})
        assert GlueUnSerializer(data).object('') == 2

    def test_default_latest_load(self):
        data = json.dumps({'': {'_type': 'glue.core.Data'}})
        assert GlueUnSerializer(data).object('') == 3


class TestVersionedDict(object):

    def test_bad_add(self):
        d = VersionedDict()
        with pytest.raises(KeyError):
            d['nonsequential', 2] = 5

    def test_get(self):
        d = VersionedDict()
        d['key', 1] = 5
        d['key', 2] = 6
        d['key', 3] = 7

        assert d['key'] == (7, 3)
        assert d.get_version('key', 1) == 5
        assert d.get_version('key', 2) == 6
        assert d.get_version('key', 3) == 7

        with pytest.raises(KeyError) as exc:
            d['missing']

    def test_contains(self):

        d = VersionedDict()
        assert 'key' not in d

        d['key', 1] = 3
        assert 'key' in d

    def test_overwrite_forbidden(self):

        d = VersionedDict()
        d['key', 1] = 3

        with pytest.raises(KeyError) as exc:
            d['key', 1] = 3

    def test_noninteger_version(self):
        d = VersionedDict()
        with pytest.raises(ValueError) as exc:
            d['key', 'bad'] = 4
