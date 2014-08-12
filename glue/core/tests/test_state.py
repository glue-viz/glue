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


def doubler(x):
    return 2 * x


class Dummy(object):
    pass


class Cloner(object):

    def __init__(self, obj):
        self.s = GlueSerializer(obj)
        self.us = GlueUnSerializer.loads(self.s.dumps())

    def get(self, o):
        return self.us.object(self.s.id(o))


class Circular(object):

    def __gluestate__(self, context):
        return dict(other=context.id(self.other))

    @classmethod
    def __setgluestate__(cls, rec, context):
        result = cls()
        yield result
        result.other = context.object(rec['other'])


def test_generator_loaders():

    f = Circular()
    b = Circular()
    f.other = b
    b.other = f

    f2 = clone(f)
    assert f2.other.other is f2


def test_none():
    assert clone(None) is None


def test_data():
    d = core.Data(x=[1, 2, 3], label='testing')
    d2 = clone(d)
    assert d2.label == 'testing'

    np.testing.assert_array_equal(d2['x'], [1, 2, 3])
    np.testing.assert_array_equal(d2['Pixel Axis 0'], [0, 1, 2])


def test_data_style():
    d = core.Data(x=[1, 2, 3])
    d.style.color = 'blue'
    d2 = clone(d)
    assert d2.style.color == 'blue'


def test_data_factory():
    with make_file(TEST_FITS_DATA, '.fits', decompress=True) as infile:
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


def test_range_subset():
    d = core.Data(x=[1, 2, 3])
    s = d.new_subset(label='range')
    s.subset_state = core.subset.RangeSubsetState(0.5, 2.5, att=d.id['x'])
    d2 = clone(d)

    np.testing.assert_array_equal(
        d2.subsets[0].to_mask(), [True, True, False])


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
        c = Cloner(app)
        copy = c.us.object('__main__')

        hub1 = app.session.hub
        hub2 = copy.session.hub

        assert len(hub1._subscriptions) == len(hub2._subscriptions)

        # data collections are the same
        for d1, d2 in zip(app.session.data_collection,
                          copy.session.data_collection):
            assert d1.label == d2.label
            for cid1, cid2 in zip(d1.components, d2.components):
                assert cid1.label == cid2.label

                # order of components unspecified if label collisions
                cid2 = c.get(cid1)
                np.testing.assert_array_almost_equal(d1[cid1, 0:1],
                                                     d2[cid2, 0:1], 3)

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
        link = core.ComponentLink([d1.id['x']], d2.id['y'], doubler)
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

        s1 = dc.new_subset_group()
        s2 = dc.new_subset_group()
        assert len(w.layers) == 3
        l1, l2, l3 = w.layers
        l1.zorder, l2.zorder = l2.zorder, l1.zorder
        l3.visible = False
        assert l3.visible is False
        copy = self.check_clone(app)
        assert copy.viewers[0][0].layers[-1].visible is False

    def test_multi_tab(self):
        d = core.Data(label='hist', x=[[1, 2], [2, 3]])
        dc = core.DataCollection([d])

        app = GlueApplication(dc)
        w1 = app.new_data_viewer(HistogramWidget, data=d)
        app.new_tab()
        w2 = app.new_data_viewer(HistogramWidget, data=d)
        assert app.viewers == ((w1,), (w2,))

        self.check_clone(app)

    def test_histogram(self):
        d = core.Data(label='hist', x=[[1, 2], [2, 3]])
        dc = core.DataCollection([d])

        app = GlueApplication(dc)
        w = app.new_data_viewer(HistogramWidget, data=d)
        self.check_clone(app)

        dc.new_subset_group()
        assert len(w.layers) == 2
        self.check_clone(app)

        w.nbins = 7
        self.check_clone(app)

    def test_subset_groups_remain_synced_after_restore(self):
        # regrssion test for 352
        d = core.Data(label='hist', x=[[1, 2], [2, 3]])
        dc = core.DataCollection([d])
        dc.new_subset_group()
        app = GlueApplication(dc)

        app2 = clone(app)
        sg = app2.data_collection.subset_groups[0]
        assert sg.style.parent is sg

        sg.style.color = '#112233'
        assert sg.subsets[0].style.color == '#112233'


class TestVersioning(object):

    def setup_method(self, method):

        @saver(Dummy, version=1)
        def s(d, context):
            return dict(v=3)

        @loader(Dummy, version=1)
        def l(d, context):
            return 3

        @saver(Dummy, version=2)
        def s(d, context):
            return dict(v=4)

        @loader(Dummy, version=2)
        def l(rec, context):
            return 4

    def teardown_method(self, method):
        GlueSerializer.dispatch._data[Dummy].pop(1)
        GlueSerializer.dispatch._data[Dummy].pop(2)
        GlueUnSerializer.dispatch._data[Dummy].pop(1)
        GlueUnSerializer.dispatch._data[Dummy].pop(2)

    def test_default_latest_save(self):
        state = GlueSerializer(Dummy()).dumpo().values()[0]
        assert state['v'] == 4
        assert state['_protocol'] == 2

    def test_legacy_load(self):
        data = json.dumps({'': {'_type': 'glue.core.tests.test_state.Dummy',
                                '_protocol': 1, 'v': 3}})
        assert GlueUnSerializer(data).object('') == 3

    def test_default_legacy_load(self):
        data = json.dumps({'': {'_type': 'glue.core.tests.test_state.Dummy'}})
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

    def test_get_missing(self):
        d = VersionedDict()
        d['key', 1] = 5
        with pytest.raises(KeyError) as exc:
            d.get_version('key', 2)
        assert exc.value.args[0] == 'No value associated with version 2 of key'

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
