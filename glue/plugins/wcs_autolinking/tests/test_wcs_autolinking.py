import pytest
import numpy as np

from astropy.wcs import WCS
from glue.core import Data, DataCollection
from glue.plugins.wcs_autolinking.wcs_autolinking import wcs_autolink, WCSLink
from glue.core.coordinates import WCSCoordinates
from glue.core.link_helpers import MultiLink
from glue.core.tests.test_state import clone
from glue.dialogs.link_editor.state import EditableLinkFunctionState

# The autolinking functionality requires the APE 14 WCS implementation in
# Astropy 3.1.
pytest.importorskip("astropy", minversion="3.1")


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
    data1.coords = WCSCoordinates(wcs=WCS(naxis=1))
    data1['x'] = [1, 2, 3]

    data2 = Data()
    data2.coords = WCSCoordinates(wcs=WCS(naxis=1))
    data2['x'] = [4, 5, 6]

    dc = DataCollection([data1, data2])
    links = wcs_autolink(dc)
    assert len(links) == 0


def test_wcs_autolink_dimensional_mismatch():

    # No links should be found because the WCS don't actually have well defined
    # physical types.

    wcs1 = WCS(naxis=1)
    wcs1.wcs.ctype = ['FREQ']
    wcs1.wcs.set()

    data1 = Data()
    data1.coords = WCSCoordinates(wcs=wcs1)
    data1['x'] = [1, 2, 3]

    wcs2 = WCS(naxis=3)
    wcs2.wcs.ctype = 'DEC--TAN', 'FREQ', 'RA---TAN'
    wcs2.wcs.set()

    data2 = Data()
    data2.coords = WCSCoordinates(wcs=wcs2)
    data2['x'] = np.ones((2, 3, 4))

    dc = DataCollection([data1, data2])
    links = wcs_autolink(dc)
    assert len(links) == 0


def test_wcs_autolink_spectral_cube():

    # This should link all coordinates

    wcs1 = WCS(naxis=3)
    wcs1.wcs.ctype = 'DEC--TAN', 'FREQ', 'RA---TAN'
    wcs1.wcs.set()

    data1 = Data()
    data1.coords = WCSCoordinates(wcs=wcs1)
    data1['x'] = np.ones((2, 3, 4))
    pz1, py1, px1 = data1.pixel_component_ids

    wcs2 = WCS(naxis=3)
    wcs2.wcs.ctype = 'GLON-CAR', 'GLAT-CAR', 'FREQ'
    wcs2.wcs.set()

    data2 = Data()
    data2.coords = WCSCoordinates(wcs=wcs2)
    data2['x'] = np.ones((2, 3, 4))
    pz2, py2, px2 = data2.pixel_component_ids

    dc = DataCollection([data1, data2])
    links = wcs_autolink(dc)
    assert len(links) == 1
    link = links[0]
    assert isinstance(link, MultiLink)
    assert len(link) == 6
    assert link[0].get_to_id() == pz2
    assert link[0].get_from_ids() == [pz1, py1, px1]
    assert link[1].get_to_id() == py2
    assert link[1].get_from_ids() == [pz1, py1, px1]
    assert link[2].get_to_id() == px2
    assert link[2].get_from_ids() == [pz1, py1, px1]
    assert link[3].get_to_id() == pz1
    assert link[3].get_from_ids() == [pz2, py2, px2]
    assert link[4].get_to_id() == py1
    assert link[4].get_from_ids() == [pz2, py2, px2]
    assert link[5].get_to_id() == px1
    assert link[5].get_from_ids() == [pz2, py2, px2]


def test_wcs_autolink_image_and_spectral_cube():

    # This should link the celestial coordinates

    wcs1 = WCS(naxis=2)
    wcs1.wcs.ctype = 'DEC--TAN', 'RA---TAN'
    wcs1.wcs.set()

    data1 = Data()
    data1.coords = WCSCoordinates(wcs=wcs1)
    data1['x'] = np.ones((2, 3))
    py1, px1 = data1.pixel_component_ids

    wcs2 = WCS(naxis=3)
    wcs2.wcs.ctype = 'GLON-CAR', 'FREQ', 'GLAT-CAR'
    wcs2.wcs.set()

    data2 = Data()
    data2.coords = WCSCoordinates(wcs=wcs2)
    data2['x'] = np.ones((2, 3, 4))
    pz2, py2, px2 = data2.pixel_component_ids

    dc = DataCollection([data1, data2])
    links = wcs_autolink(dc)
    assert len(links) == 1
    link = links[0]
    assert isinstance(link, MultiLink)
    assert len(link) == 4
    assert link[0].get_to_id() == px2
    assert link[0].get_from_ids() == [py1, px1]
    assert link[1].get_to_id() == pz2
    assert link[1].get_from_ids() == [py1, px1]
    assert link[2].get_to_id() == py1
    assert link[2].get_from_ids() == [px2, pz2]
    assert link[3].get_to_id() == px1
    assert link[3].get_from_ids() == [px2, pz2]


def test_clone_wcs_link():

    # Make sure that WCSLink can be serialized/deserialized

    wcs1 = WCS(naxis=2)
    wcs1.wcs.ctype = 'DEC--TAN', 'RA---TAN'
    wcs1.wcs.set()

    data1 = Data(label='Data 1')
    data1.coords = WCSCoordinates(wcs=wcs1)
    data1['x'] = np.ones((2, 3))

    wcs2 = WCS(naxis=3)
    wcs2.wcs.ctype = 'GLON-CAR', 'FREQ', 'GLAT-CAR'
    wcs2.wcs.set()

    data2 = Data(label='Data 2')
    data2.coords = WCSCoordinates(wcs=wcs2)
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
    data1.coords = WCSCoordinates(wcs=wcs1)
    data1['x'] = np.ones((2, 3))

    wcs2 = WCS(naxis=3)
    wcs2.wcs.ctype = 'GLON-CAR', 'FREQ', 'GLAT-CAR'
    wcs2.wcs.set()

    data2 = Data(label='Data 2')
    data2.coords = WCSCoordinates(wcs=wcs2)
    data2['x'] = np.ones((2, 3, 4))

    link1 = WCSLink(data1, data2)

    link2 = EditableLinkFunctionState(link1).link

    assert isinstance(link2, WCSLink)
    assert link2.data1.label == 'Data 1'
    assert link2.data2.label == 'Data 2'
