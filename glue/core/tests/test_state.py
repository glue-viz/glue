from __future__ import absolute_import, division, print_function

import json
from io import BytesIO

import pytest
import numpy as np
from numpy.testing import assert_equal

from glue.external import six
from glue import core
from glue.core.component import CategoricalComponent, DateTimeComponent
from glue.tests.helpers import requires_astropy, make_file

from ..data_factories import load_data
from ..data_factories.tests.test_fits import TEST_FITS_DATA
from ..state import (GlueSerializer, GlueUnSerializer,
                     saver, loader, VersionedDict)


def clone(object, include_data=False):
    gs = GlueSerializer(object, include_data=include_data)
    oid = gs.id(object)
    dump = gs.dumps()
    gu = GlueUnSerializer.loads(dump)
    result = gu.object(oid)
    return result


def doubler(x):
    return 2 * x


def containers_equal(c1, c2):
    """Check that two container-like items have the same contents,
    ignoring differences relating to the type of container
    """
    if isinstance(c1, six.string_types):
        return c1 == c2

    try:
        for a, b in zip(c1, c2):
            if not containers_equal(a, b):
                return False
            if isinstance(c1, dict) and isinstance(c2, dict):
                if not containers_equal(c1[a], c2[b]):
                    return False
    except TypeError:
        pass

    return True


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
    np.testing.assert_array_equal(d2['Pixel Axis 0 [x]'], [0, 1, 2])


def test_data_style():
    d = core.Data(x=[1, 2, 3])
    d.style.color = 'blue'
    d2 = clone(d)
    assert d2.style.color == 'blue'


@requires_astropy
def test_data_factory():
    with make_file(TEST_FITS_DATA, '.fits', decompress=True) as infile:
        d = load_data(infile)
        d2 = clone(d)

    np.testing.assert_array_equal(d['PRIMARY'], d2['PRIMARY'])


@requires_astropy
def test_data_factory_include_data():
    with make_file(TEST_FITS_DATA, '.fits', decompress=True) as infile:
        d = load_data(infile)
        d2 = clone(d, include_data=True)

    np.testing.assert_array_equal(d['PRIMARY'], d2['PRIMARY'])


def test_save_numpy_scalar():
    assert clone(np.float32(5)) == 5


@requires_astropy
def tests_data_factory_double():
    # ensure double-cloning doesn't somehow lose lightweight references
    from astropy.io import fits
    d = np.random.normal(0, 1, (100, 100, 100))
    s = BytesIO()
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
    assert_equal(r2.vx, [0, 0, 1])
    assert_equal(r2.vy, [0, 1, 0])


def test_projected3d_roi():
    roi_2d = core.roi.PolygonalROI(vx=[0.5, 2.5, 2.5, 0.5], vy=[1, 1, 3.5, 3.5])
    roi = core.roi.Projected3dROI(roi_2d=roi_2d, projection_matrix=np.eye(4))
    roi_clone = clone(roi)
    x = [1, 2, 3]
    y = [2, 3, 4]
    z = [5, 6, 7]
    assert roi.contains(x, y).tolist() == roi_clone.contains(x, y).tolist()


def test_matplotlib_cmap():
    from matplotlib import cm
    assert clone(cm.gist_heat) is cm.gist_heat


def test_binary_component_link():
    d1 = core.Data(x=[1, 2, 3])
    d1['y'] = d1.id['x'] + 1
    assert_equal(d1['y'], [2, 3, 4])
    d2 = clone(d1)
    assert_equal(d2['y'], [2, 3, 4])


class Spam(object):
    pass


@saver(Spam)
def _save_spam(spam, context):
    return {'spam': spam}


@loader(Spam)
def _load_spam(rec, context):
    pass


def test_no_circular():

    # If a saver/loader is implemented incorrectly, this used to lead to
    # non-informative circular references, so we check that the error message
    # is more useful now

    with pytest.raises(TypeError) as exc:
        clone(Spam())
    assert "is not JSON serializable" in exc.value.args[0]


def test_categorical_component():
    c = CategoricalComponent(['a', 'b', 'c', 'a', 'b'], categories=['a', 'b', 'c'])
    c2 = clone(c)
    assert isinstance(c2, CategoricalComponent)
    np.testing.assert_array_equal(c.codes, [0, 1, 2, 0, 1])
    np.testing.assert_array_equal(c.labels, ['a', 'b', 'c', 'a', 'b'])
    np.testing.assert_array_equal(c.categories, ['a', 'b', 'c'])


def test_datetime_component():
    c = DateTimeComponent(np.array([100, 200, 300], dtype='M8[D]'))
    c2 = clone(c)
    assert isinstance(c2, DateTimeComponent)
    np.testing.assert_array_equal(c.data, c2.data)
    assert isinstance(c2.data[0], np.datetime64)


class DummyClass(object):
    pass


class TestVersioning(object):

    def setup_method(self, method):

        @saver(DummyClass, version=1)
        def s1(d, context):
            return dict(v=3)

        @loader(DummyClass, version=1)
        def l1(d, context):
            return 3

        @saver(DummyClass, version=2)
        def s2(d, context):
            return dict(v=4)

        @loader(DummyClass, version=2)
        def l2(rec, context):
            return 4

    def teardown_method(self, method):
        GlueSerializer.dispatch._data[DummyClass].pop(1)
        GlueSerializer.dispatch._data[DummyClass].pop(2)
        GlueUnSerializer.dispatch._data[DummyClass].pop(1)
        GlueUnSerializer.dispatch._data[DummyClass].pop(2)

    def test_default_latest_save(self):
        assert list(GlueSerializer(DummyClass()).dumpo().values())[0]['v'] == 4
        assert list(GlueSerializer(DummyClass()).dumpo().values())[0]['_protocol'] == 2

    def test_legacy_load(self):
        data = json.dumps({'': {'_type': 'glue.core.tests.test_state.DummyClass',
                                '_protocol': 1, 'v': 2}})
        assert GlueUnSerializer(data).object('') == 3

    def test_default_earliest_load(self):
        data = json.dumps({'': {'_type': 'glue.core.tests.test_state.DummyClass'}})
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
