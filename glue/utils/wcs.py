from __future__ import absolute_import, division, print_function

import numpy as np

__all__ = ['axis_correlation_matrix']


def axis_correlation_matrix(wcs):

    n_world = len(wcs.wcs.ctype)
    n_pixel = wcs.naxis

    # If there are any distortions present, we assume that there may be
    # correlations between all axes. Maybe if some distortions only apply
    # to the image plane we can improve this
    for distortion_attribute in ('sip', 'det2im1', 'det2im2'):
        if getattr(wcs, distortion_attribute):
            return np.ones((n_world, n_pixel), dtype=bool)

    # Assuming linear world coordinates along each axis, the correlation
    # matrix would be given by whether or not the PC matrix is zero
    matrix = wcs.wcs.get_pc() != 0

    # We now need to check specifically for celestial coordinates since
    # these can assume correlations because of spherical distortions. For
    # each celestial coordinate we copy over the pixel dependencies from
    # the other celestial coordinates.
    celestial = (wcs.wcs.axis_types // 1000) % 10 == 2
    celestial_indices = np.nonzero(celestial)[0]
    for world1 in celestial_indices:
        for world2 in celestial_indices:
            if world1 != world2:
                matrix[world1] |= matrix[world2]
                matrix[world2] |= matrix[world1]

    return matrix
