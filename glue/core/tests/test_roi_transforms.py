import numpy as np
from numpy.testing import assert_allclose

from glue.core.roi_pretransforms import ProjectionMplTransform
from glue.core.state import GlueSerializer, GlueUnSerializer
from glue.tests.helpers import xfail_matplotlib_lt_37


def roundtrip_transform(transform):
    gs = GlueSerializer(transform)
    out_str = gs.dumps()
    obj = GlueUnSerializer.loads(out_str)
    return obj.object('__main__')


def test_simple_polar_mpl_transform():
    angles = np.deg2rad(np.array([0, 45, 90, 180, 300, 810, 0, 0]))
    radii = np.array([0, 5, 4, 1, 0, 2, -10, 10])
    transform = ProjectionMplTransform('polar', [0, 2 * np.pi], [-5, 5], 'linear', 'linear')
    x, y = transform(angles, radii)
    expected_x = np.array([0.75, 0.8535533905932736, 0.5, 0.2, .625, 0.5, np.nan, 1.25])
    expected_y = np.array([0.5, 0.8535533905932736, 0.95, 0.5, 0.28349364905389035,
                           0.85, np.nan, 0.5])
    assert_allclose(x, expected_x)
    assert_allclose(y, expected_y)
    new_transform = roundtrip_transform(transform)
    new_x, new_y = new_transform(angles, radii)
    assert_allclose(new_x, x, rtol=1e-14)
    assert_allclose(new_y, y, rtol=1e-14)


@xfail_matplotlib_lt_37
def test_log_polar_mpl_transform():
    angles = np.deg2rad(np.array([0, 90, 180]))
    radii = np.array([10, 100, 1000])
    transform = ProjectionMplTransform('polar', [0, 2 * np.pi], [1, 10000], 'linear', 'log')
    x, y = transform(angles, radii)
    expected_x = np.array([0.625, 0.5, 0.125])
    expected_y = np.array([0.5, 0.75, 0.5])
    assert_allclose(x, expected_x)
    assert_allclose(y, expected_y)
    new_transform = roundtrip_transform(transform)
    new_x, new_y = new_transform(angles, radii)
    assert_allclose(new_x, x, rtol=1e-14)
    assert_allclose(new_y, y, rtol=1e-14)


def test_wedge_polar_mpl_transform():
    angles = np.deg2rad(np.array([0, 45, 90, 135, 180]))
    radii = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    transform = ProjectionMplTransform('polar', [0, np.pi], [0, 0.5], 'linear', 'linear')
    x, y = transform(angles, radii)
    assert_allclose(x, np.array([0.6, 0.64142136, 0.5, 0.21715729, 0]))
    # For just the upper half, y is between 0.25 and 0.75
    assert_allclose(y, np. array([0.25, 0.39142136, 0.55, 0.53284271, 0.25]))
    new_transform = roundtrip_transform(transform)
    new_x, new_y = new_transform(angles, radii)
    assert_allclose(new_x, x, rtol=1e-14)
    assert_allclose(new_y, y, rtol=1e-14)


def test_aitoff_mpl_transform():
    transform = ProjectionMplTransform('aitoff', [-np.pi, np.pi],
                                       [-np.pi / 2, np.pi / 2], 'linear', 'linear')
    long = np.deg2rad(np.array([0, -90, 0, 45]))
    lat = np.deg2rad(np.array([0, 0, 45, -45]))
    x, y = transform(long, lat)
    expected_x = np.array([0.5, 0.25, 0.5, 0.59771208])
    expected_y = np.array([0.5, 0.5, 0.75, 0.24466602])
    assert_allclose(x, expected_x)
    assert_allclose(y, expected_y)
    new_transform = roundtrip_transform(transform)
    new_x, new_y = new_transform(long, lat)
    assert_allclose(new_x, x, rtol=1e-14)
    assert_allclose(new_y, y, rtol=1e-14)


def test_hammer_mpl_transform():
    transform = ProjectionMplTransform('hammer', [-np.pi, np.pi],
                                       [-np.pi / 2, np.pi / 2], 'linear', 'linear')
    long = np.deg2rad(np.array([0, -90, 0, 45]))
    lat = np.deg2rad(np.array([0, 0, 45, -45]))
    x, y = transform(long, lat)
    expected_x = np.array([0.5, 0.22940195, 0.5, 0.60522557])
    expected_y = np.array([0.5, 0.5, 0.77059805, 0.22503235])
    assert_allclose(x, expected_x)
    assert_allclose(y, expected_y)
    new_transform = roundtrip_transform(transform)
    new_x, new_y = new_transform(long, lat)
    assert_allclose(new_x, x, rtol=1e-14)
    assert_allclose(new_y, y, rtol=1e-14)


def test_lambert_mpl_projection():
    transform = ProjectionMplTransform('lambert', [-np.pi, np.pi],
                                       [-np.pi / 2, np.pi / 2], 'linear', 'linear')
    long = np.deg2rad(np.array([0, -90, 0, 45]))
    lat = np.deg2rad(np.array([0, 0, 45, -45]))
    x, y = transform(long, lat)
    expected_x = np.array([0.5, 0.14644661, 0.5, 0.64433757])
    expected_y = np.array([0.5, 0.5, 0.69134172, 0.29587585])
    assert_allclose(x, expected_x)
    assert_allclose(y, expected_y)
    new_transform = roundtrip_transform(transform)
    new_x, new_y = new_transform(long, lat)
    assert_allclose(new_x, x, rtol=1e-14)
    assert_allclose(new_y, y, rtol=1e-14)


def test_mollweide_mpl_projection():
    transform = ProjectionMplTransform('mollweide', [-np.pi, np.pi],
                                       [-np.pi / 2, np.pi / 2], 'linear', 'linear')
    long = np.deg2rad(np.array([0, -90, 0, 45]))
    lat = np.deg2rad(np.array([0, 0, 45, -45]))
    x, y = transform(long, lat)
    expected_x = np.array([0.5, 0.25, 0.5, 0.60076564])
    expected_y = np.array([0.5, 0.5, 0.79587254, 0.20412746])
    assert_allclose(x, expected_x)
    assert_allclose(y, expected_y)
    new_transform = roundtrip_transform(transform)
    new_x, new_y = new_transform(long, lat)
    assert_allclose(new_x, x, rtol=1e-14)
    assert_allclose(new_y, y, rtol=1e-14)
