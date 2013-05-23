# This file is adapted from aplpy/wcs_util.py.

# APLpy - The Astronomical Plotting Library in Python
# Copyright (c) 2010 Thomas P. Robitaille and Eli Bressert
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import numpy as np

# Galactic conversion constants
RA_NGP = np.radians(192.859508333333)
DEC_NGP = np.radians(27.1283361111111)
L_CP = np.radians(122.932)
L_0 = L_CP - np.pi / 2.
RA_0 = RA_NGP + np.pi / 2.
DEC_0 = np.pi / 2. - DEC_NGP


def fk52gal(ra, dec):
    """ Convert from equatorial to galactic coordinates

    :param ra: scalar or array. Right Ascension
    :param dec: scalar or array. Declination

    :rtype: Tuple of L,B coordinates for each input
    """
    ra, dec = np.radians(ra), np.radians(dec)

    sinb = np.sin(dec) * np.cos(DEC_0) - np.cos(dec) * np.sin(ra - RA_0) \
        * np.sin(DEC_0)

    b = np.arcsin(sinb)

    cosl = np.cos(dec) * np.cos(ra - RA_0) / np.cos(b)
    sinl = (np.sin(dec) * np.sin(DEC_0) + np.cos(dec) * np.sin(ra - RA_0) *
            np.cos(DEC_0)) / np.cos(b)

    b = np.degrees(b)

    l = np.arccos(cosl)
    l[np.where(sinl < 0.)] = - l[np.where(sinl < 0.)]

    l = np.degrees(l + L_0)

    l = np.mod(l, 360.)
    b = np.mod(b + 90., 180.) - 90.

    return l, b


def gal2fk5(l, b):
    """ Convert from galactic to equatorial coordinates

    :param l: scalar or array. Galactic longitude
    :param b: scalar or array. Galactic latitude

    :rtype: Tuple of RA, dec coordinates for each input
    """

    l = np.radians(l)
    b = np.radians(b)

    sind = np.sin(b) * np.sin(DEC_NGP) + \
        np.cos(b) * np.cos(DEC_NGP) * np.sin(l - L_0)

    dec = np.arcsin(sind)

    cosa = np.cos(l - L_0) * np.cos(b) / np.cos(dec)
    sina = (np.cos(b) * np.sin(DEC_NGP) * np.sin(l - L_0) -
            np.sin(b) * np.cos(DEC_NGP)) / np.cos(dec)

    dec = np.degrees(dec)

    ra = np.arccos(cosa)
    ra[np.where(sina < 0.)] = -ra[np.where(sina < 0.)]

    ra = np.degrees(ra + RA_0)

    ra = np.mod(ra, 360.)
    dec = np.mod(dec + 90., 180.) - 90.

    return ra, dec
