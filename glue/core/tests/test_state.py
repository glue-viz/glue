import numpy as np
import json

from ..state import GlueSerializer, GlueUnSerializer
from ... import core


def clone(object):
    gs = GlueSerializer()
    oid = gs.id(object)
    state = json.dumps(gs.do_all())
    gu = GlueUnSerializer(state)
    return gu.object(oid)


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
