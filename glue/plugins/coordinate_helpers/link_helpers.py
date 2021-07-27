# A plugin to enable link helpers for Astronomical coordinate conversions.

from astropy import units as u
from astropy.coordinates import (ICRS, FK5, FK4, Galactic, Galactocentric,
                                 galactocentric_frame_defaults)

from glue.core.link_helpers import BaseMultiLink
from glue.config import link_helper


__all__ = ["BaseCelestialMultiLink", "Galactic_to_FK5", "FK4_to_FK5",
           "ICRS_to_FK5", "Galactic_to_FK4", "ICRS_to_FK4",
           "ICRS_to_Galactic"]


class BaseCelestialMultiLink(BaseMultiLink):

    display = None
    frame_in = None
    frame_out = None

    def forwards(self, in_lon, in_lat):
        cin = self.frame_in(in_lon * u.deg, in_lat * u.deg)
        cout = cin.transform_to(self.frame_out)
        return cout.spherical.lon.degree, cout.spherical.lat.degree

    def backwards(self, out_lon, out_lat):
        cout = self.frame_out(out_lon * u.deg, out_lat * u.deg)
        cin = cout.transform_to(self.frame_in)
        return cin.spherical.lon.degree, cin.spherical.lat.degree

    # Backward-compatibility with glue-core <0.15
    forward = forwards
    backward = backwards


@link_helper(category='Astronomy')
class Galactic_to_FK5(BaseCelestialMultiLink):
    description = 'Link Galactic and FK5 (J2000) Equatorial coordinates'
    labels1 = 'l', 'b'
    labels2 = 'ra (fk5)', 'dec (fk5)'
    display = "Galactic <-> FK5 (J2000)"
    frame_in = Galactic
    frame_out = FK5


@link_helper(category='Astronomy')
class FK4_to_FK5(BaseCelestialMultiLink):
    description = 'Link FK4 (B1950) and FK5 (J2000) Equatorial coordinates'
    labels1 = 'ra (fk4)', 'dec (fk4)'
    labels2 = 'ra (fk5)', 'dec (fk5)'
    display = "FK4 (B1950) <-> FK5 (J2000)"
    frame_in = FK4
    frame_out = FK5


@link_helper(category='Astronomy')
class ICRS_to_FK5(BaseCelestialMultiLink):
    description = 'Link ICRS and FK5 (J2000) Equatorial coordinates'
    labels1 = 'ra (fk4)', 'dec (fk4)'
    labels2 = 'ra (icrs)', 'dec (icrs)'
    display = "ICRS <-> FK5 (J2000)"
    frame_in = ICRS
    frame_out = FK5


@link_helper(category='Astronomy')
class Galactic_to_FK4(BaseCelestialMultiLink):
    description = 'Link Galactic and FK4 (B1950) Equatorial coordinates'
    labels1 = 'l', 'b'
    labels2 = 'ra (fk4)', 'dec (fk4)'
    display = "Galactic <-> FK4 (B1950)"
    frame_in = Galactic
    frame_out = FK4


@link_helper(category='Astronomy')
class ICRS_to_FK4(BaseCelestialMultiLink):
    description = 'Link ICRS and FK4 (B1950) Equatorial coordinates'
    labels1 = 'ra (icrs)', 'dec (icrs)'
    labels2 = 'ra (fk4)', 'dec (fk4)'
    display = "ICRS <-> FK4 (B1950)"
    frame_in = ICRS
    frame_out = FK4


@link_helper(category='Astronomy')
class ICRS_to_Galactic(BaseCelestialMultiLink):
    description = 'Link ICRS and Galactic coordinates'
    labels1 = 'ra (icrs)', 'dec (icrs)'
    labels2 = 'l', 'b'
    display = "ICRS <-> Galactic"
    frame_in = ICRS
    frame_out = Galactic


@link_helper(category='Astronomy')
class GalactocentricToGalactic(BaseMultiLink):

    description = 'Link 3D Galactocentric and Galactic coordinates'
    labels1 = 'x (kpc)', 'y (kpc)', 'z (kpc)'
    labels2 = 'l (deg)', 'b (deg)', 'distance (kpc)'
    display = "3D Galactocentric <-> Galactic"

    def forwards(self, x_kpc, y_kpc, z_kpc):
        with galactocentric_frame_defaults.set('pre-v4.0'):
            gal = Galactocentric(x=x_kpc * u.kpc, y=y_kpc * u.kpc, z=z_kpc * u.kpc).transform_to(Galactic())
        return gal.l.degree, gal.b.degree, gal.distance.to(u.kpc).value

    def backwards(self, l_deg, b_deg, d_kpc):
        with galactocentric_frame_defaults.set('pre-v4.0'):
            gal = Galactic(l=l_deg * u.deg, b=b_deg * u.deg, distance=d_kpc * u.kpc).transform_to(Galactocentric())
        return gal.x.to(u.kpc).value, gal.y.to(u.kpc).value, gal.z.to(u.kpc).value
