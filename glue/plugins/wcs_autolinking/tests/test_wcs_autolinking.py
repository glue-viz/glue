import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.wcs import WCS
from astropy.wcs.wcsapi import BaseLowLevelWCS

from glue.core import Data, DataCollection
from glue.plugins.wcs_autolinking.wcs_autolinking import (wcs_autolink, WCSLink,
                                                          IncompatibleWCS,
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


def test_wcs_autolink_2D_emptywcs():

    # No links should be found because the WCS don't actually have well defined
    # physical types.

    data1 = Data()
    data1.coords = WCS(naxis=2)
    data1['x'] = [[1, 2, 3]]

    data2 = Data()
    data2.coords = WCS(naxis=2)
    data2['x'] = [[4, 5, 6]]

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
    pz2, _py2, px2 = data2.pixel_component_ids

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


@pytest.mark.parametrize('physical_types',
                         [['SPAM', 'FREQ'], ['SPAM', 'EGGS'], ['WAVE', ''], ['', '']])
def test_celestial_with_unknown_axes(physical_types):

    # Regression test for a bug that caused n-d datasets with celestial axes
    # and axes with unknown physical types to not even be linked by celestial
    # axes. Also testing various corner cases with one or both datasets having
    # unknown or no physical type.

    wcs1 = WCS(naxis=3)
    wcs1.wcs.ctype = 'DEC--TAN', 'RA---TAN', physical_types[0]
    wcs1.wcs.set()

    data1 = Data()
    data1.coords = wcs1
    data1['x'] = np.ones((2, 3, 4))
    _pz1, py1, px1 = data1.pixel_component_ids

    wcs2 = WCS(naxis=3)
    wcs2.wcs.ctype = 'GLON-CAR', physical_types[1], 'GLAT-CAR'
    wcs2.wcs.set()

    data2 = Data()
    data2.coords = wcs2
    data2['x'] = np.ones((2, 3, 4))
    pz2, _py2, px2 = data2.pixel_component_ids

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
    pw1, pz1, py1, _px1 = data1.pixel_component_ids

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


# Seems only to fail under py310
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
    py1, _px1 = data1.pixel_component_ids

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


# Seems no longer to fail with supported Python versions (Astropy 4.1+)
#@pytest.mark.xfail
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
    _pz1, py1, _px1 = data1.pixel_component_ids

    wcs2 = WCS(naxis=2)
    wcs2.wcs.ctype = 'FREQ', 'TIME'
    wcs2.wcs.set()

    data2 = Data(label='Data 2')
    data2.coords = wcs2
    data2['x'] = np.ones((4, 5))
    _py2, px2 = data2.pixel_component_ids

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


class SimpleLowLevelWCS(BaseLowLevelWCS):
    """
    A minimal APE-14 low-level-only WCS (not a BaseHighLevelWCS), with
    world = scale * pixel along each axis.
    """

    def __init__(self, scales, physical_types, units):
        self._scales = scales
        self._physical_types = list(physical_types)
        self._units = list(units)

    @property
    def pixel_n_dim(self):
        return len(self._scales)

    @property
    def world_n_dim(self):
        return len(self._scales)

    @property
    def world_axis_physical_types(self):
        return self._physical_types

    @property
    def world_axis_units(self):
        return self._units

    @property
    def axis_correlation_matrix(self):
        return np.identity(self.world_n_dim, dtype=bool)

    def pixel_to_world_values(self, *pixel):
        world = [np.asarray(p) * s for p, s in zip(pixel, self._scales)]
        return world[0] if self.world_n_dim == 1 else tuple(world)

    def world_to_pixel_values(self, *world):
        pixel = [np.asarray(w) / s for w, s in zip(world, self._scales)]
        return pixel[0] if self.pixel_n_dim == 1 else tuple(pixel)

    @property
    def world_axis_object_components(self):
        return [(f'world{i}', 0, 'value') for i in range(self.world_n_dim)]

    @property
    def world_axis_object_classes(self):
        return {f'world{i}': (u.Quantity, (), {'unit': unit})
                for i, unit in enumerate(self._units)}


class CoupledLowLevelWCS(SimpleLowLevelWCS):
    """
    A 3D low-level WCS with a non-trivial axis_correlation_matrix: the two
    celestial world axes each depend on the same two pixel axes (as for
    celestial -TAB axes), and the axis order is not simply reversed.
    """

    def __init__(self):
        super().__init__((1, 1, 1),
                         ['custom:pos.helioprojective.lon',
                          'custom:pos.helioprojective.lat',
                          'time'],
                         ['arcsec', 'arcsec', 's'])

    @property
    def axis_correlation_matrix(self):
        return np.array([[False, True, True],
                         [False, True, True],
                         [True, False, False]])

    def pixel_to_world_values(self, p0, p1, p2):
        return (np.asarray(p1) + np.asarray(p2),
                np.asarray(p1) - np.asarray(p2),
                np.asarray(p0))

    def world_to_pixel_values(self, lon, lat, time):
        return (np.asarray(time),
                (np.asarray(lon) + np.asarray(lat)) / 2,
                (np.asarray(lon) - np.asarray(lat)) / 2)


class BrokenLowLevelWCS(SimpleLowLevelWCS):
    """A low-level WCS whose world-to-pixel transform always fails."""

    def world_to_pixel_values(self, *world):
        raise ValueError("cannot transform")


HPC_TYPES = ['custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat']


def _low_level_data(label, wcs, shape):
    data = Data(label=label)
    data.coords = wcs
    data['x'] = np.ones(shape)
    return data


def test_wcs_autolink_low_level():

    # Datasets whose coords only implement the low-level APE-14 API should
    # be linked through their matching physical types.

    data1 = _low_level_data('Data 1', SimpleLowLevelWCS((1, 2), HPC_TYPES, ['arcsec'] * 2), (3, 4))
    data2 = _low_level_data('Data 2', SimpleLowLevelWCS((2, 4), HPC_TYPES, ['arcsec'] * 2), (3, 4))

    dc = DataCollection([data1, data2])
    links = wcs_autolink(dc)
    assert len(links) == 1
    link = links[0]
    assert len(link) == 4

    # world = scale * pixel, so pixel2 = pixel1 * scale1 / scale2
    assert_allclose(link.forwards(3.0, 8.0), (1.5, 4.0))
    assert_allclose(link.backwards(1.5, 4.0), (3.0, 8.0))


def test_wcs_autolink_low_level_and_fits_wcs():

    # A low-level-only WCS should also link to an astropy WCS.

    data1 = _low_level_data('Data 1', SimpleLowLevelWCS((2,), ['em.freq'], ['Hz']), 3)

    wcs2 = WCS(naxis=1)
    wcs2.wcs.ctype = ['FREQ']
    wcs2.wcs.cunit = ['Hz']
    wcs2.wcs.set()

    data2 = _low_level_data('Data 2', wcs2, 3)

    dc = DataCollection([data1, data2])
    links = wcs_autolink(dc)
    assert len(links) == 1
    link = links[0]
    assert len(link) == 2

    # low-level pixel p -> 2 * p Hz; astropy world = pixel + 1, so pixel = 2 * p - 1
    assert_allclose(link.forwards(3.0), 5.0)
    assert_allclose(link.backwards(5.0), 3.0)


def test_wcs_autolink_low_level_disjoint():

    # No matching physical types: no link and no exception.

    data1 = _low_level_data('Data 1', SimpleLowLevelWCS((1,), ['custom:spam'], ['']), 3)
    data2 = _low_level_data('Data 2', SimpleLowLevelWCS((1, 1), ['custom:eggs', 'custom:ham'], ['', '']), (3, 4))

    dc = DataCollection([data1, data2])
    assert wcs_autolink(dc) == []

    with pytest.raises(IncompatibleWCS):
        WCSLink(data1, data2)


def test_wcs_autolink_low_level_transform_failure():

    # Matching physical types whose world objects cannot actually be
    # transformed (e.g. helioprojective frames where one WCS carries no
    # observer) should skip cleanly, never traceback.

    data1 = _low_level_data('Data 1', SimpleLowLevelWCS((1, 1), HPC_TYPES, ['arcsec'] * 2), (3, 4))
    data2 = _low_level_data('Data 2', BrokenLowLevelWCS((1, 1), HPC_TYPES, ['arcsec'] * 2), (3, 4))

    dc = DataCollection([data1, data2])
    assert wcs_autolink(dc) == []

    with pytest.raises(IncompatibleWCS):
        WCSLink(data1, data2)


def test_wcs_autolink_correlated_axes():

    # The pixel axes to keep must come from axis_correlation_matrix, not from
    # assuming world axis i maps to pixel axis world_n_dim - i - 1. Here the
    # two matched world axes are both coupled to WCS pixel axes 1 and 2, i.e.
    # numpy axes 1 and 0, while the old assumption would have kept 2 and 1.

    data1 = _low_level_data('Data 1', CoupledLowLevelWCS(), (2, 3, 4))
    data2 = _low_level_data('Data 2', SimpleLowLevelWCS((1, 1), HPC_TYPES, ['arcsec'] * 2), (3, 4))

    dc = DataCollection([data1, data2])
    links = wcs_autolink(dc)
    assert len(links) == 1
    link = links[0]
    assert len(link) == 4

    assert [cid.axis for cid in link.cids1] == [1, 0]
    assert [cid.axis for cid in link.cids2] == [1, 0]

    # lon = p1 + p2, lat = p1 - p2 and the second dataset has world == pixel
    assert_allclose(link.forwards(3.0, 1.0), (4.0, 2.0))
    assert_allclose(link.backwards(4.0, 2.0), (3.0, 1.0))


class FullyCoupledLowLevelWCS(CoupledLowLevelWCS):
    """A 3D low-level WCS whose celestial world axes depend on all pixel axes."""

    @property
    def axis_correlation_matrix(self):
        return np.array([[True, True, True],
                         [True, True, True],
                         [True, False, False]])

    def pixel_to_world_values(self, p0, p1, p2):
        p0, p1, p2 = np.asarray(p0), np.asarray(p1), np.asarray(p2)
        return p1 + p2 + p0, p1 - p2 + p0, p0

    def world_to_pixel_values(self, lon, lat, time):
        lon, lat, time = np.asarray(lon), np.asarray(lat), np.asarray(time)
        return time, (lon + lat) / 2 - time, (lon - lat) / 2


def test_wcs_autolink_low_level_disjoint_same_ndim():

    # Same-dimensionality low-level pairs with disjoint physical types must
    # not be linked positionally by the transform probe (regression test:
    # wrapped low-level WCSes produce plain Quantities, so the probe cannot
    # detect the type mismatch and must not be trusted for them).

    data1 = _low_level_data('Data 1', SimpleLowLevelWCS((1,), ['custom:spam'], ['']), 3)
    data2 = _low_level_data('Data 2', SimpleLowLevelWCS((2,), ['custom:eggs'], ['']), 3)

    dc = DataCollection([data1, data2])
    assert wcs_autolink(dc) == []

    with pytest.raises(IncompatibleWCS):
        WCSLink(data1, data2)


def test_wcs_autolink_low_level_world_order_mismatch():

    # Matched world axes in different orders on the two sides would be
    # silently transposed by positional Quantity pairing, so such low-level
    # pairs are refused.

    data1 = _low_level_data('Data 1', SimpleLowLevelWCS((1, 1), HPC_TYPES, ['arcsec'] * 2), (3, 4))
    data2 = _low_level_data('Data 2', SimpleLowLevelWCS((1, 1), HPC_TYPES[::-1], ['arcsec'] * 2), (3, 4))

    dc = DataCollection([data1, data2])
    assert wcs_autolink(dc) == []

    with pytest.raises(IncompatibleWCS):
        WCSLink(data1, data2)


def test_wcs_autolink_ndim_mismatch():

    # A dataset whose array dimensionality does not match its WCS must be
    # skipped cleanly without breaking autolinking of the other datasets.

    data1 = _low_level_data('Data 1', SimpleLowLevelWCS((1, 1, 1),
                                                        HPC_TYPES + ['time'],
                                                        ['arcsec', 'arcsec', 's']), (3, 4))
    data2 = _low_level_data('Data 2', SimpleLowLevelWCS((1, 1), HPC_TYPES, ['arcsec'] * 2), (3, 4))
    data3 = _low_level_data('Data 3', SimpleLowLevelWCS((2, 2), HPC_TYPES, ['arcsec'] * 2), (3, 4))

    dc = DataCollection([data1, data2, data3])
    links = wcs_autolink(dc)
    assert len(links) == 1
    assert {links[0].data1, links[0].data2} == {data2, data3}


def test_wcs_autolink_low_level_11d():

    # Regression test for lexical sorting of the kept numpy axes, which
    # misorders single- vs double-digit axis indices for ndim >= 11.

    types1 = [None] * 11
    types1[0] = 'em.wl'    # WCS pixel axis 0 -> numpy axis 10
    types1[8] = 'time'     # WCS pixel axis 8 -> numpy axis 2
    scales1 = [1.0] * 11
    scales1[0] = 2.0
    scales1[8] = 3.0

    shape1 = [1] * 11
    shape1[10], shape1[2] = 3, 4

    data1 = _low_level_data('Data 1', SimpleLowLevelWCS(scales1, types1, [''] * 11), tuple(shape1))
    data2 = _low_level_data('Data 2', SimpleLowLevelWCS((1, 1), ['em.wl', 'time'], ['', '']), (3, 4))

    dc = DataCollection([data1, data2])
    links = wcs_autolink(dc)
    assert len(links) == 1
    link = links[0]

    assert [cid.axis for cid in link.cids1] == [10, 2]
    assert [cid.axis for cid in link.cids2] == [1, 0]

    # world = scale * pixel on both sides
    assert_allclose(link.forwards(3.0, 1.0), (6.0, 3.0))
    assert_allclose(link.backwards(6.0, 3.0), (3.0, 1.0))


def test_wcs_autolink_all_axes_coupled():

    # A 3D WCS whose matched world axes are coupled to every pixel axis
    # against a 2D dataset: the sliced WCSes end up with incompatible world
    # objects, which should be skipped cleanly (regression test for an
    # IndexError caused by slicing_axes1/slicing_axes2 aliasing one list).

    data1 = _low_level_data('Data 1', FullyCoupledLowLevelWCS(), (2, 3, 4))
    data2 = _low_level_data('Data 2', SimpleLowLevelWCS((1, 1), HPC_TYPES, ['arcsec'] * 2), (3, 4))

    dc = DataCollection([data1, data2])
    assert wcs_autolink(dc) == []

    with pytest.raises(IncompatibleWCS):
        WCSLink(data1, data2)


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
    x5, _y5 = offset_link.backwards(x1, y1)

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
    x5, _y5 = affine_link.backwards(x1, y1)

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


def test_no_wcs_overlap():

    wcs1 = WCS(naxis=2)
    wcs1.wcs.ctype = 'RA---TAN', 'DEC--TAN'
    wcs1.wcs.crval = 10, 20
    wcs1.wcs.set()

    data1 = Data(label='Data 1')
    data1.coords = wcs1
    data1['x'] = np.ones((2, 3))

    wcs2 = WCS(naxis=2)
    wcs2.wcs.ctype = 'RA---TAN', 'DEC--TAN'
    wcs2.wcs.crval = 190, -20
    wcs2.wcs.set()

    data2 = Data(label='Data 2')
    data2.coords = wcs2
    data2['x'] = np.ones((2, 3))

    link = WCSLink(data1, data2)

    with pytest.raises(NoAffineApproximation, match='no overlap'):
        link.as_affine_link()
