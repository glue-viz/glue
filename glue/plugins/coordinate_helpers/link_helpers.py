# A plugin to enable link helpers for Astronomical coordinate conversions.
# This requires Astropy >= 0.4.

# Coordinate transforms (requires Astropy>)

from ...core.config import link_function, link_helper
from ...core.link_helpers import MultiLink

from astropy import units as u
from astropy.coordinates import ICRS, FK5, FK4, Galactic

__all__ = ["BaseCelestialMultiLink", "Galactic_to_FK5", "FK4_to_FK5",
           "ICRS_to_FK5", "Galactic_to_FK4", "ICRS_to_FK4",
           "ICRS_to_Galactic"]


class BaseCelestialMultiLink(MultiLink):

    display = None
    frame_in = None
    frame_out = None

    def __init__(self, in_lon, in_lat, out_lon, out_lat):
        MultiLink.__init__(self, [in_lon, in_lat],
                                 [out_lon, out_lat],
                                 self.forward, self.backward)

    def forward(self, in_lon, in_lat):
        c = self.frame_in(in_lon * u.deg, in_lat * u.deg)
        out = c.transform_to(self.frame_out)
        return out.spherical.lon.degree, out.spherical.lat.degree

    def backward(self, in_lon, in_lat):
        c = self.frame_out(in_lon * u.deg, in_lat * u.deg)
        out = c.transform_to(self.frame_in)
        return out.spherical.lon.degree, out.spherical.lat.degree


@link_helper('Link Galactic and FK5 (J2000) Equatorial coordinates',
             input_labels=['l', 'b', 'ra (fk5)', 'dec (fk5)'])
class Galactic_to_FK5(BaseCelestialMultiLink):
    display = "Celestial Coordinates: Galactic <-> FK5 (J2000)"
    frame_in = Galactic
    frame_out = FK5


@link_helper('Link FK4 (B1950) and FK5 (J2000) Equatorial coordinates',
             input_labels=['ra (fk4)', 'dec (fk4)', 'ra (fk5)', 'dec (fk5)'])
class FK4_to_FK5(BaseCelestialMultiLink):
    display = "Celestial Coordinates: FK4 (B1950) <-> FK5 (J2000)"
    frame_in = FK4
    frame_out = FK5


@link_helper('Link ICRS and FK5 (J2000) Equatorial coordinates',
             input_labels=['ra (icrs)', 'dec (icrs)', 'ra (fk5)', 'dec (fk5)'])
class ICRS_to_FK5(BaseCelestialMultiLink):
    display = "Celestial Coordinates: ICRS <-> FK5 (J2000)"
    frame_in = ICRS
    frame_out = FK5

@link_helper('Link Galactic and FK4 (B1950) Equatorial coordinates',
             input_labels=['l', 'b', 'ra (fk4)', 'dec (fk4)'])
class Galactic_to_FK4(BaseCelestialMultiLink):
    display = "Celestial Coordinates: Galactic <-> FK4 (B1950)"
    frame_in = Galactic
    frame_out = FK4


@link_helper('Link ICRS and FK4 (B1950) Equatorial coordinates',
             input_labels=['ra (icrs)', 'dec (icrs)', 'ra (fk4)', 'dec (fk4)'])
class ICRS_to_FK4(BaseCelestialMultiLink):
    display = "Celestial Coordinates: ICRS <-> FK4 (B1950)"
    frame_in = ICRS
    frame_out = FK4


@link_helper('Link ICRS and Galactic coordinates',
             input_labels=['ra (icrs)', 'dec (icrs)', 'l', 'b'])
class ICRS_to_Galactic(BaseCelestialMultiLink):
    display = "Celestial Coordinates: ICRS <-> Galactic"
    frame_in = ICRS
    frame_out = Galactic
