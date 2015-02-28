# A plugin to enable link helpers for Astronomical coordinate conversions.
# This requires Astropy >= 0.4.

# Coordinate transforms (requires Astropy>)

from ...config import link_function, link_helper
from ...core.link_helpers import MultiLink

from astropy import units as u
from astropy.coordinates import FK5, Galactic


def fk52gal(lon, lat):
    c = FK5(lon * u.deg, lat * u.deg)
    g = c.transform_to(Galactic)
    return g.l.degree, g.b.degree


def gal2fk5(lon, lat):
    g = Galactic(lon * u.deg, lat * u.deg)
    c = g.transform_to(FK5)
    return c.ra.degree, c.dec.degree


@link_helper('Link Galactic and Equatorial coordinates',
             input_labels=['l', 'b', 'ra', 'dec'])
class Galactic2Equatorial(MultiLink):

    """
    Instantiate a ComponentList with four ComponentLinks that map galactic
    and equatorial coordinates

    :param l: ComponentID for galactic longitude
    :param b: ComponentID for galactic latitude
    :param ra: ComponentID for J2000 Right Ascension
    :param dec: ComponentID for J2000 Declination

    Returns a :class:`LinkCollection` object which links
    these ComponentIDs
    """

    def __init__(self, l, b, ra, dec):
        MultiLink.__init__(self, [ra, dec], [l, b], fk52gal, gal2fk5)


@link_function(info="", output_labels=['l'])
def radec2glon(ra, dec):
    """
    Compute galactic longitude from right ascension and declination.
    """
    return fk52gal(ra, dec)[0]


@link_function(info="", output_labels=['b'])
def radec2glat(ra, dec):
    """
    Compute galactic latitude from right ascension and declination.
    """
    return fk52gal(ra, dec)[1]


@link_function(info="", output_labels=['ra'])
def lb2ra(lon, lat):
    """
    Compute right ascension from galactic longitude and latitude.
    """
    return gal2fk5(lon, lat)[0]


@link_function(info="", output_labels=['dec'])
def lb2dec(lon, lat):
    """
    Compute declination from galactic longitude and latitude.
    """
    return gal2fk5(lon, lat)[1]
