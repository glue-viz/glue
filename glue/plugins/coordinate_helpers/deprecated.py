from astropy import units as u
from astropy.coordinates import FK5, Galactic

from ...core.link_helpers import MultiLink


def fk52gal(ra, dec):
    c = FK5(ra * u.deg, dec * u.deg)
    out = c.transform_to(Galactic)
    return out.l.degree, out.b.degree


def gal2fk5(l, b):
    c = Galactic(l * u.deg, b * u.deg)
    out = c.transform_to(FK5)
    return out.ra.degree, out.dec.degree



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


def radec2glon(ra, dec):
    """
    Compute galactic longitude from right ascension and declination.
    """
    return fk52gal(ra, dec)[0]


def radec2glat(ra, dec):
    """
    Compute galactic latitude from right ascension and declination.
    """
    return fk52gal(ra, dec)[1]


def lb2ra(lon, lat):
    """
    Compute right ascension from galactic longitude and latitude.
    """
    return gal2fk5(lon, lat)[0]


def lb2dec(lon, lat):
    """
    Compute declination from galactic longitude and latitude.
    """
    return gal2fk5(lon, lat)[1]
