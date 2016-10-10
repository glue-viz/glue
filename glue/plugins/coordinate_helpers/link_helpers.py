# A plugin to enable link helpers for Astronomical coordinate conversions.
# This requires Astropy >= 0.4.

# Coordinate transforms (requires Astropy>)

from __future__ import absolute_import, division, print_function

from astropy import units as u
from astropy.coordinates import ICRS, FK5, FK4, Galactic, Galactocentric

from glue.core.link_helpers import MultiLink
from glue.config import link_helper


__all__ = ["BaseCelestialMultiLink", "Galactic_to_FK5", "FK4_to_FK5",
           "ICRS_to_FK5", "Galactic_to_FK4", "ICRS_to_FK4",
           "ICRS_to_Galactic"]


class BaseCelestialMultiLink(MultiLink):

    display = None
    frame_in = None
    frame_out = None

    def __init__(self, in_lon, in_lat, out_lon, out_lat):
        super(BaseCelestialMultiLink, self).__init__(in_lon, in_lat, out_lon, out_lat)
        self.create_links([in_lon, in_lat], [out_lon, out_lat],
                          forwards=self.forward, backwards=self.backward)

    def forward(self, in_lon, in_lat):
        c = self.frame_in(in_lon * u.deg, in_lat * u.deg)
        out = c.transform_to(self.frame_out)
        return out.spherical.lon.degree, out.spherical.lat.degree

    def backward(self, in_lon, in_lat):
        c = self.frame_out(in_lon * u.deg, in_lat * u.deg)
        out = c.transform_to(self.frame_in)
        return out.spherical.lon.degree, out.spherical.lat.degree


@link_helper('Link Galactic and FK5 (J2000) Equatorial coordinates',
             input_labels=['l', 'b', 'ra (fk5)', 'dec (fk5)'],
             category='Astronomy')
class Galactic_to_FK5(BaseCelestialMultiLink):
    display = "Galactic <-> FK5 (J2000)"
    frame_in = Galactic
    frame_out = FK5


@link_helper('Link FK4 (B1950) and FK5 (J2000) Equatorial coordinates',
             input_labels=['ra (fk4)', 'dec (fk4)', 'ra (fk5)', 'dec (fk5)'],
             category='Astronomy')
class FK4_to_FK5(BaseCelestialMultiLink):
    display = "FK4 (B1950) <-> FK5 (J2000)"
    frame_in = FK4
    frame_out = FK5


@link_helper('Link ICRS and FK5 (J2000) Equatorial coordinates',
             input_labels=['ra (icrs)', 'dec (icrs)', 'ra (fk5)', 'dec (fk5)'],
             category='Astronomy')
class ICRS_to_FK5(BaseCelestialMultiLink):
    display = "ICRS <-> FK5 (J2000)"
    frame_in = ICRS
    frame_out = FK5


@link_helper('Link Galactic and FK4 (B1950) Equatorial coordinates',
             input_labels=['l', 'b', 'ra (fk4)', 'dec (fk4)'],
             category='Astronomy')
class Galactic_to_FK4(BaseCelestialMultiLink):
    display = "Galactic <-> FK4 (B1950)"
    frame_in = Galactic
    frame_out = FK4


@link_helper('Link ICRS and FK4 (B1950) Equatorial coordinates',
             input_labels=['ra (icrs)', 'dec (icrs)', 'ra (fk4)', 'dec (fk4)'],
             category='Astronomy')
class ICRS_to_FK4(BaseCelestialMultiLink):
    display = "ICRS <-> FK4 (B1950)"
    frame_in = ICRS
    frame_out = FK4


@link_helper('Link ICRS and Galactic coordinates',
             input_labels=['ra (icrs)', 'dec (icrs)', 'l', 'b'],
             category='Astronomy')
class ICRS_to_Galactic(BaseCelestialMultiLink):
    display = "ICRS <-> Galactic"
    frame_in = ICRS
    frame_out = Galactic


@link_helper('Link 3D Galactocentric and Galactic coordinates',
             input_labels=['x (kpc)', 'y (kpc)', 'z (kpc)', 'l (deg)', 'b (deg)', 'distance (kpc)'],
             category='Astronomy')
class GalactocentricToGalactic(MultiLink):

    display = "3D Galactocentric <-> Galactic"

    def __init__(self, x_id, y_id, z_id, l_id, b_id, d_id):
        super(GalactocentricToGalactic, self).__init__(x_id, y_id, z_id, l_id, b_id, d_id)
        self.create_links([x_id, y_id, z_id], [l_id, b_id, d_id],
                          self.forward, self.backward)

    def forward(self, x_kpc, y_kpc, z_kpc):
        gal = Galactocentric(x=x_kpc * u.kpc, y=y_kpc * u.kpc, z=z_kpc * u.kpc).transform_to(Galactic)
        return gal.l.degree, gal.b.degree, gal.distance.to(u.kpc).value

    def backward(self, l_deg, b_deg, d_kpc):
        gal = Galactic(l=l_deg * u.deg, b=b_deg * u.deg, distance=d_kpc * u.kpc).transform_to(Galactocentric)
        return gal.x.to(u.kpc).value, gal.y.to(u.kpc).value, gal.z.to(u.kpc).value
