import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.wcs import WCS

from glue.core import Data, DataCollection
from glue.plugins.wcs_autolinking.wcs_autolinking import (wcs_autolink, WCSLink,
                                                          OffsetLink, AffineLink,
                                                          NoAffineApproximation)
from glue.core.link_helpers import MultiLink
from glue.core.tests.test_state import clone
from glue.dialogs.link_editor.state import EditableLinkFunctionState


def test_wcs_autolink_nowcs():

    # No links should be found because there are no WCS coordinates present

    data1 = Data(x=[1, 2, 3])
    data2 = Data(x=[4, 5, 6])
    dc = DataCollection([data1, data2])
    links = wcs_autolink(dc)
    assert len(links) == 0


def test_wcs_autolink_emptywcs():

    # No links should be found because the WCS don't actually have well defined
    # physical types.

    data1 = Data()
    data1.coords = WCS(naxis=1)
    data1['x'] = [1, 2, 3]

    data2 = Data()
    data2.coords = WCS(naxis=1)
    data2['x'] = [4, 5, 6]

    dc = DataCollection([data1, data2])
    links = wcs_autolink(dc)
    assert len(links) == 0


def test_wcs_autolink_spectral_cube():

    # This should link all coordinates

    wcs1 = WCS(naxis=3)
    wcs1.wcs.ctype = 'DEC--TAN', 'FREQ', 'RA---TAN'
    wcs1.wcs.set()

    data1 = Data()
    data1.coords = wcs1
    data1['x'] = np.ones((2, 3, 4))
    pz1, py1, px1 = data1.pixel_component_ids

    wcs2 = WCS(naxis=3)
    wcs2.wcs.ctype = 'GLON-CAR', 'GLAT-CAR', 'FREQ'
    wcs2.wcs.set()

    data2 = Data()
    data2.coords = wcs2
    data2['x'] = np.ones((2, 3, 4))
    pz2, py2, px2 = data2.pixel_component_ids

    dc = DataCollection([data1, data2])
    links = wcs_autolink(dc)
    assert len(links) == 1
    link = links[0]
    assert isinstance(link, MultiLink)
    assert len(link) == 6
    assert link[0].get_to_id() == px2
    assert link[0].get_from_ids() == [px1, py1, pz1]
    assert link[1].get_to_id() == py2
    assert link[1].get_from_ids() == [px1, py1, pz1]
    assert link[2].get_to_id() == pz2
    assert link[2].get_from_ids() == [px1, py1, pz1]
    assert link[3].get_to_id() == px1
    assert link[3].get_from_ids() == [px2, py2, pz2]
    assert link[4].get_to_id() == py1
    assert link[4].get_from_ids() == [px2, py2, pz2]
    assert link[5].get_to_id() == pz1
    assert link[5].get_from_ids() == [px2, py2, pz2]


def test_wcs_autolink_image_and_spectral_cube():

    # This should link the celestial coordinates

    wcs1 = WCS(naxis=2)
    wcs1.wcs.ctype = 'DEC--TAN', 'RA---TAN'
    wcs1.wcs.set()

    data1 = Data()
    data1.coords = wcs1
    data1['x'] = np.ones((2, 3))
    py1, px1 = data1.pixel_component_ids

    wcs2 = WCS(naxis=3)
    wcs2.wcs.ctype = 'GLON-CAR', 'FREQ', 'GLAT-CAR'
    wcs2.wcs.set()

    data2 = Data()
    data2.coords = wcs2
    data2['x'] = np.ones((2, 3, 4))
    pz2, py2, px2 = data2.pixel_component_ids

    dc = DataCollection([data1, data2])
    links = wcs_autolink(dc)
    assert len(links) == 1
    link = links[0]
    assert isinstance(link, MultiLink)
    assert len(link) == 4
    assert link[0].get_to_id() == px2
    assert link[0].get_from_ids() == [px1, py1]
    assert link[1].get_to_id() == pz2
    assert link[1].get_from_ids() == [px1, py1]
    assert link[2].get_to_id() == px1
    assert link[2].get_from_ids() == [px2, pz2]
    assert link[3].get_to_id() == py1
    assert link[3].get_from_ids() == [px2, pz2]


def test_clone_wcs_link():

    # Make sure that WCSLink can be serialized/deserialized

    wcs1 = WCS(naxis=2)
    wcs1.wcs.ctype = 'DEC--TAN', 'RA---TAN'
    wcs1.wcs.set()

    data1 = Data(label='Data 1')
    data1.coords = wcs1
    data1['x'] = np.ones((2, 3))

    wcs2 = WCS(naxis=3)
    wcs2.wcs.ctype = 'GLON-CAR', 'FREQ', 'GLAT-CAR'
    wcs2.wcs.set()

    data2 = Data(label='Data 2')
    data2.coords = wcs2
    data2['x'] = np.ones((2, 3, 4))

    link1 = WCSLink(data1, data2)
    link2 = clone(link1)

    assert isinstance(link2, WCSLink)
    assert link2.data1.label == 'Data 1'
    assert link2.data2.label == 'Data 2'


def test_link_editor():

    # Make sure that the WCSLink works property in the link editor and is
    # returned unmodified. The main way to check that is just to make sure that
    # the link round-trips when going through EditableLinkFunctionState.

    wcs1 = WCS(naxis=2)
    wcs1.wcs.ctype = 'DEC--TAN', 'RA---TAN'
    wcs1.wcs.set()

    data1 = Data(label='Data 1')
    data1.coords = wcs1
    data1['x'] = np.ones((2, 3))

    wcs2 = WCS(naxis=3)
    wcs2.wcs.ctype = 'GLON-CAR', 'FREQ', 'GLAT-CAR'
    wcs2.wcs.set()

    data2 = Data(label='Data 2')
    data2.coords = wcs2
    data2['x'] = np.ones((2, 3, 4))

    link1 = WCSLink(data1, data2)

    link2 = EditableLinkFunctionState(link1).link

    assert isinstance(link2, WCSLink)
    assert link2.data1.label == 'Data 1'
    assert link2.data2.label == 'Data 2'


def test_celestial_with_unknown_axes():

    # Regression test for a bug that caused n-d datasets with celestial axes
    # and axes with unknown physical types to not even be linked by celestial
    # axes.

    wcs1 = WCS(naxis=3)
    wcs1.wcs.ctype = 'DEC--TAN', 'RA---TAN', 'SPAM'
    wcs1.wcs.set()

    data1 = Data()
    data1.coords = wcs1
    data1['x'] = np.ones((2, 3, 4))
    pz1, py1, px1 = data1.pixel_component_ids

    wcs2 = WCS(naxis=3)
    wcs2.wcs.ctype = 'GLON-CAR', 'FREQ', 'GLAT-CAR'
    wcs2.wcs.set()

    data2 = Data()
    data2.coords = wcs2
    data2['x'] = np.ones((2, 3, 4))
    pz2, py2, px2 = data2.pixel_component_ids

    dc = DataCollection([data1, data2])
    links = wcs_autolink(dc)
    assert len(links) == 1
    link = links[0]
    assert isinstance(link, MultiLink)
    assert len(link) == 4
    assert link[0].get_to_id() == px2
    assert link[0].get_from_ids() == [px1, py1]
    assert link[1].get_to_id() == pz2
    assert link[1].get_from_ids() == [px1, py1]
    assert link[2].get_to_id() == px1
    assert link[2].get_from_ids() == [px2, pz2]
    assert link[3].get_to_id() == py1
    assert link[3].get_from_ids() == [px2, pz2]


def test_wcs_autolinking_of_2d_cube_with_temporal_and_spectral_axes_case_1():
    """
    A test to confirm that two 2D data cubes with matching number of dimensions
    where the first is temporal and the next one spectral (vacuum wavelength in this
    case) is indeed autolinked.
    """

    wcs1 = WCS(naxis=2)
    wcs1.wcs.ctype = 'TIME', 'WAVE'
    wcs1.wcs.set()

    data1 = Data(label='Data 1')
    data1.coords = wcs1
    data1['x'] = np.ones((2, 3))
    py1, px1 = data1.pixel_component_ids

    wcs2 = WCS(naxis=2)
    wcs2.wcs.ctype = 'TAI', 'WAVE'
    wcs2.wcs.set()

    data2 = Data(label='Data 2')
    data2.coords = wcs2
    data2['x'] = np.ones((2, 3))
    py2, px2 = data2.pixel_component_ids

    dc = DataCollection([data1, data2])
    links = wcs_autolink(dc)
    assert len(links) == 1
    link = links[0]
    assert isinstance(link, MultiLink)
    assert len(link) == 4
    assert link[0].get_to_id() == px2
    assert link[0].get_from_ids() == [px1, py1]
    assert link[1].get_to_id() == py2
    assert link[1].get_from_ids() == [px1, py1]
    assert link[2].get_to_id() == px1
    assert link[2].get_from_ids() == [px2, py2]
    assert link[3].get_to_id() == py1
    assert link[3].get_from_ids() == [px2, py2]


def test_wcs_autolinking_of_2d_cube_with_temporal_and_spectral_axes_case_2():
    """
    A test to confirm that two 2D data cubes with matching number of dimensions
    where the one is spectral (air wavelength in this case) and the other one
    temporal is indeed autolinked, to test that the order does not matter.
    """

    wcs1 = WCS(naxis=2)
    wcs1.wcs.ctype = 'AWAV', 'TIME'
    wcs1.wcs.set()

    data1 = Data(label='Data 1')
    data1.coords = wcs1
    data1['x'] = np.ones((2, 3))
    py1, px1 = data1.pixel_component_ids

    wcs2 = WCS(naxis=2)
    wcs2.wcs.ctype = 'TIME', 'AWAV'
    wcs2.wcs.set()

    data2 = Data(label='Data 2')
    data2.coords = wcs2
    data2['x'] = np.ones((2, 3))
    py2, px2 = data2.pixel_component_ids

    dc = DataCollection([data1, data2])
    links = wcs_autolink(dc)
    assert len(links) == 1
    link = links[0]
    assert isinstance(link, MultiLink)
    assert len(link) == 4
    assert link[0].get_to_id() == px2
    assert link[0].get_from_ids() == [px1, py1]
    assert link[1].get_to_id() == py2
    assert link[1].get_from_ids() == [px1, py1]
    assert link[2].get_to_id() == px1
    assert link[2].get_from_ids() == [px2, py2]
    assert link[3].get_to_id() == py1
    assert link[3].get_from_ids() == [px2, py2]


def test_has_celestial_with_time_and_spectral_axes():
    """
    To test the case in which we have two data cubes with unequal
    number of dimensions, but both have celestial axes.
    """

    wcs1 = WCS(naxis=4)
    wcs1.wcs.ctype = 'WAVE', 'HPLT-TAN', 'HPLN-TAN', 'TIME'
    wcs1.wcs.set()

    data1 = Data(label='Data 1')
    data1.coords = wcs1
    data1['x'] = np.ones((2, 3, 4, 5))
    pw1, pz1, py1, px1 = data1.pixel_component_ids

    wcs2 = WCS(naxis=3)
    wcs2.wcs.ctype = 'HPLN-TAN', 'HPLT-TAN', 'TIME'
    wcs2.wcs.set()

    data2 = Data(label='Data 2')
    data2.coords = wcs2
    data2['x'] = np.ones((2, 3, 4))
    pz2, py2, px2 = data2.pixel_component_ids

    dc = DataCollection([data1, data2])
    links = wcs_autolink(dc)
    assert len(links) == 1
    link = links[0]
    assert isinstance(link, MultiLink)
    assert len(link) == 6
    assert link[0].get_to_id() == px2
    assert link[0].get_from_ids() == [py1, pz1, pw1]
    assert link[1].get_to_id() == py2
    assert link[1].get_from_ids() == [py1, pz1, pw1]
    assert link[2].get_to_id() == pz2
    assert link[2].get_from_ids() == [py1, pz1, pw1]
    assert link[3].get_to_id() == py1
    assert link[3].get_from_ids() == [px2, py2, pz2]
    assert link[4].get_to_id() == pz1
    assert link[4].get_from_ids() == [px2, py2, pz2]
    assert link[5].get_to_id() == pw1
    assert link[5].get_from_ids() == [px2, py2, pz2]


@pytest.mark.xfail
def test_2d_and_1d_data_cubes_with_no_celestial_axes():
    """
    Test the case where we have one 2D dataset with WAVE and TIME
    as CTYPEs and a 1D dataset with WAVE as the CTYPE.
    """

    wcs1 = WCS(naxis=2)
    wcs1.wcs.ctype = 'TIME', 'WAVE'
    wcs1.wcs.set()

    data1 = Data(label='Data 1')
    data1.coords = wcs1
    data1['x'] = np.ones((2, 3))
    py1, px1 = data1.pixel_component_ids

    wcs2 = WCS(naxis=1)
    wcs2.wcs.ctype = ['WAVE']
    wcs2.wcs.set()

    data2 = Data(label='Data 2')
    data2.coords = wcs2
    data2['x'] = np.ones(3)
    px2 = data2.pixel_component_ids

    dc = DataCollection([data1, data2])
    links = wcs_autolink(dc)
    assert len(links) == 1
    link = links[0]
    assert isinstance(link, MultiLink)
    assert len(link) == 2
    assert ' '.join(str(link[0].get_to_id()).split()[:2]) == ' '.join(str(py1).split()[:2])
    assert ' '.join(str(link[0].get_from_ids()).split()[:2]) == ' '.join(str(px2).split()[:2])


@pytest.mark.xfail
def test_link_of_spectral_axes_of_different_physical_types():
    """
    To check that there is no auto-link of spectral axes of two different physical types, e.g.
    between FREQ and WAVE.
    """

    wcs1 = WCS(naxis=1)
    wcs1.wcs.ctype = ['FREQ']
    wcs1.wcs.set()

    data1 = Data(label='Data 1')
    data1.coords = wcs1
    data1['x'] = np.ones(2)
    px1 = data1.pixel_component_ids

    wcs2 = WCS(naxis=1)
    wcs2.wcs.ctype = ['WAVE']
    wcs2.wcs.set()

    data2 = Data(label='Data 2')
    data2.coords = wcs2
    data2['x'] = np.ones(2)
    px2 = data2.pixel_component_ids

    dc = DataCollection([data1, data2])
    links = wcs_autolink(dc)
    assert len(links) == 1
    link = links[0]
    assert isinstance(link, MultiLink)
    assert len(link) == 2
    assert link[0].get_to_id() == str(px1[0])
    assert str(link[0].get_from_ids()) == str(px2)
    assert link[1].get_to_id() == str(px2)
    assert str(link[1].get_from_ids()) == str(px1)


def test_cube_has_celestial_and_cube_without_celestial_axes_1():
    """
    To test that there should be a link between a 3D dataset with celestial axes
    and a 2D dataset with no celestial axes (variant 1).
    """

    wcs1 = WCS(naxis=3)
    wcs1.wcs.ctype = 'RA---TAN', 'FREQ', 'DEC--TAN'
    wcs1.wcs.set()

    data1 = Data(label='Data 1')
    data1.coords = wcs1
    data1['x'] = np.ones((2, 3, 4))
    pz1, py1, px1 = data1.pixel_component_ids

    wcs2 = WCS(naxis=2)
    wcs2.wcs.ctype = 'FREQ', 'TIME'
    wcs2.wcs.set()

    data2 = Data(label='Data 2')
    data2.coords = wcs2
    data2['x'] = np.ones((4, 5))
    py2, px2 = data2.pixel_component_ids

    dc = DataCollection([data1, data2])
    links = wcs_autolink(dc)
    assert len(links) == 1
    link = links[0]
    assert isinstance(link, MultiLink)
    assert len(link) == 2
    assert link[0].get_to_id() == px2
    assert link[0].get_from_ids() == [py1]
    assert link[1].get_to_id() == py1
    assert link[1].get_from_ids() == [px2]


@pytest.mark.xfail
def test_cube_has_celestial_and_cube_without_celestial_axes_2():
    """
    To test that there should be a link between a 3D dataset with celestial axes
    and a 2D dataset with no celestial axes (variant 2).
    TODO: To modify code base so that the FREQ axis would be linked up with the WAVE axis.
    """

    wcs1 = WCS(naxis=3)
    wcs1.wcs.ctype = 'RA---TAN', 'FREQ', 'DEC--TAN'
    wcs1.wcs.set()

    data1 = Data(label='Data 1')
    data1.coords = wcs1
    data1['x'] = np.ones((2, 3, 4))

    wcs2 = WCS(naxis=2)
    wcs2.wcs.ctype = 'WAVE', 'TIME'
    wcs2.wcs.set()

    data2 = Data(label='Data 2')
    data2.coords = wcs2
    data2['x'] = np.ones((4, 5))

    dc = DataCollection([data1, data2])
    links = wcs_autolink(dc)
    assert len(links) == 1


def test_wcs_offset_approximation():

    wcs1 = WCS(naxis=2)
    wcs1.wcs.ctype = 'DEC--TAN', 'RA---TAN'
    wcs1.wcs.set()

    data1 = Data(label='Data 1')
    data1.coords = wcs1
    data1['x'] = np.ones((2, 3))

    wcs2 = WCS(naxis=2)
    wcs2.wcs.ctype = 'DEC--TAN', 'RA---TAN'
    wcs2.wcs.crpix = -3, 5
    wcs2.wcs.set()

    data2 = Data(label='Data 2')
    data2.coords = wcs2
    data2['x'] = np.ones((2, 3))

    link = WCSLink(data1, data2)

    offset_link = link.as_affine_link(tolerance=0.1)

    assert isinstance(offset_link, OffsetLink)
    assert_allclose(offset_link.offsets, [3, -5])

    x1 = np.array([1.4, 3.2, 2.5])
    y1 = np.array([0.2, 4.3, 2.2])

    x2, y2 = link.forwards(x1, y1)
    x3, y3 = offset_link.forwards(x1, y1)

    assert_allclose(x2, x3, atol=1e-5)
    assert_allclose(y2, y3, atol=1e-5)

    x4, y4 = link.backwards(x1, y1)
    x5, y5 = offset_link.backwards(x1, y1)

    assert_allclose(x4, x5, atol=1e-5)
    assert_allclose(y4, y4, atol=1e-5)


def test_wcs_affine_approximation():

    wcs1 = WCS(naxis=2)
    wcs1.wcs.ctype = 'DEC--TAN', 'RA---TAN'
    wcs1.wcs.set()

    data1 = Data(label='Data 1')
    data1.coords = wcs1
    data1['x'] = np.ones((2, 3))

    wcs2 = WCS(naxis=2)
    wcs2.wcs.ctype = 'DEC--TAN', 'RA---TAN'
    wcs2.wcs.crpix = -3, 5
    wcs2.wcs.cd = [[2, -1], [1, 2]]
    wcs2.wcs.set()

    data2 = Data(label='Data 2')
    data2.coords = wcs2
    data2['x'] = np.ones((2, 3))

    link = WCSLink(data1, data2)

    affine_link = link.as_affine_link(tolerance=0.1)

    assert isinstance(affine_link, AffineLink)
    assert_allclose(affine_link.matrix, [[0.4, 0.2, -3.4], [-0.2, 0.4, 4.2], [0, 0, 1]], atol=1e-5)

    x1 = np.array([1.4, 3.2, 2.5])
    y1 = np.array([0.2, 4.3, 2.2])

    x2, y2 = link.forwards(x1, y1)
    x3, y3 = affine_link.forwards(x1, y1)

    assert_allclose(x2, x3, atol=1e-5)
    assert_allclose(y2, y3, atol=1e-5)

    x4, y4 = link.backwards(x1, y1)
    x5, y5 = affine_link.backwards(x1, y1)

    assert_allclose(x4, x5, atol=1e-5)
    assert_allclose(y4, y4, atol=1e-5)


def test_wcs_no_approximation():

    wcs1 = WCS(naxis=2)
    wcs1.wcs.ctype = 'DEC--TAN', 'RA---TAN'
    wcs1.wcs.set()

    data1 = Data(label='Data 1')
    data1.coords = wcs1
    data1['x'] = np.ones((2, 3))

    wcs2 = WCS(naxis=2)
    wcs2.wcs.ctype = 'DEC--TAN', 'RA---TAN'
    wcs2.wcs.crval = 30, 50
    wcs2.wcs.set()

    data2 = Data(label='Data 2')
    data2.coords = wcs2
    data2['x'] = np.ones((2, 3))

    link = WCSLink(data1, data2)

    with pytest.raises(NoAffineApproximation):
        link.as_affine_link(tolerance=0.1)
