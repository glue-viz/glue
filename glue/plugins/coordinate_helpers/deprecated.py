from __future__ import absolute_import, division, print_function

from astropy import units as u
from astropy.coordinates import FK5, Galactic


def fk52gal(ra, dec):
    c = FK5(ra * u.deg, dec * u.deg)
    out = c.transform_to(Galactic)
    return out.l.degree, out.b.degree


def gal2fk5(l, b):
    c = Galactic(l * u.deg, b * u.deg)
    out = c.transform_to(FK5)
    return out.ra.degree, out.dec.degree


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
