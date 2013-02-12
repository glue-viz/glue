"""
Interface to import astronomy specific libraries.

Since the astronomy community is currently migrating towards astropy,
there are several packages that provide ~identical functionality (i.e.
legacy libraries like pyfits, and their equivalent submodule in
astropy)

This module provides a transparent interface that defaults to astropy,
but falls back to legacy libraries if astropy isn't installed on the
users' system
"""

try:
    from astropy.io import fits
except ImportError:
    import pyfits as fits

try:
    from astropy.wcs import WCS
except ImportError:
    from pywcs import WCS
    #update interface
    WCS.all_pix2world = WCS.all_pix2sky
