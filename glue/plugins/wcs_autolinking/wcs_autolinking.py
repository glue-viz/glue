import copy

import numpy as np
from astropy import units as u
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_pixel
from astropy.wcs.wcsapi import (BaseHighLevelWCS, BaseLowLevelWCS,
                                SlicedLowLevelWCS, HighLevelWCSWrapper)
from scipy.optimize import leastsq
from glue.config import autolinker, link_helper
from glue.core.link_helpers import MultiLink


__all__ = ['IncompatibleWCS', 'WCSLink', 'wcs_autolink', 'AffineLink', 'OffsetLink',
           'NoAffineApproximation']


class NoAffineApproximation(Exception):
    pass


class OffsetLink(MultiLink):

    def __init__(self, data1=None, data2=None, cids1=None, cids2=None, offsets=None):

        self.offsets = offsets

        self.data1 = data1
        self.data2 = data2

        super().__init__(cids1, cids2, forwards=self.forwards, backwards=self.backwards)

    def forwards(self, *pixel_in):
        return tuple([pi - o for (pi, o) in zip(pixel_in, self.offsets)])

    def backwards(self, *pixel_out):
        return tuple([po + o for (po, o) in zip(pixel_out, self.offsets)])


class AffineLink(MultiLink):

    def __init__(self, data1=None, data2=None, cids1=None, cids2=None, matrix=None):

        if matrix.ndim != 2:
            raise ValueError("Affine matrix should be two-dimensional")

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Affine matrix should be square")

        if np.any(matrix[-1, :-1] != 0) or matrix[-1, -1] != 1:
            raise ValueError("Last row of matrix should be zeros and a one")

        self._matrix = matrix
        self._matrix_inv = np.linalg.inv(matrix)

        self.data1 = data1
        self.data2 = data2

        super().__init__(cids1, cids2, forwards=self.forwards, backwards=self.backwards)

    @property
    def matrix(self):
        return self._matrix

    def forwards(self, *pixel_in):
        pixel_in = np.array(np.broadcast_arrays(*(list(pixel_in) + [np.ones(np.shape(pixel_in[0]))])))
        pixel_in = np.moveaxis(pixel_in, 0, -1)
        pixel_out = np.matmul(pixel_in, self._matrix.T)
        return tuple(np.moveaxis(pixel_out, -1, 0))[:-1]

    def backwards(self, *pixel_out):
        pixel_out = np.array(np.broadcast_arrays(*(list(pixel_out) + [np.ones(np.shape(pixel_out[0]))])))
        pixel_out = np.moveaxis(pixel_out, 0, -1)
        pixel_in = np.matmul(pixel_out, self._matrix_inv.T)
        return tuple(np.moveaxis(pixel_in, -1, 0))[:-1]


class IncompatibleWCS(Exception):
    pass


def get_cids_and_functions(wcs1, wcs2, pixel_cids1, pixel_cids2,
                           forwards=None, backwards=None):

    if forwards is None:
        def forwards(*pixel_input):
            return pixel_to_pixel(wcs1, wcs2, *pixel_input)

    if backwards is None:
        def backwards(*pixel_input):
            return pixel_to_pixel(wcs2, wcs1, *pixel_input)

    pixel_input = [0] * len(pixel_cids1)

    try:
        # the case with wcs linkages
        forwards(*pixel_input)
        backwards(*pixel_input)
    except Exception:
        # the case without wcs linkages
        return None, None, None, None

    return pixel_cids1, pixel_cids2, forwards, backwards


def kept_numpy_axes(wcs_ll, matched_world_axes):
    """
    The numpy axes to keep when slicing ``wcs_ll`` down to its matched world
    axes: every pixel axis correlated with a matched world axis, except that
    pixel axes also driving unmatched world axes are sliced away as long as
    every matched world axis remains supported by another pixel axis (e.g.
    the time axis of an SJI cube, which the celestial axes are weakly
    coupled to - the link is then exact at the first exposure).
    """
    matrix = wcs_ll.axis_correlation_matrix
    keep = set()
    for world_axis in matched_world_axes:
        keep |= {int(pixel_axis) for pixel_axis in np.nonzero(matrix[world_axis])[0]}
    unmatched = [w for w in range(wcs_ll.world_n_dim) if w not in matched_world_axes]
    for pixel_axis in sorted(keep):
        if any(matrix[w, pixel_axis] for w in unmatched):
            trial = keep - {pixel_axis}
            if trial and all(any(matrix[w, p] for p in trial) for w in matched_world_axes):
                keep = trial
    # pixel axes are in x-fastest order, numpy axes are reversed
    return {int(wcs_ll.pixel_n_dim - 1 - pixel_axis) for pixel_axis in keep}


def permuted_values_functions(wcs1, wcs2):
    """
    Pixel-to-pixel transform functions for two sliced low-level WCSes whose
    matched world axes have the same physical types in a different order,
    going through world *values* with an explicit type-matched permutation
    (and unit conversion). pixel_to_pixel cannot be used for these: wrapped
    low-level WCSes produce plain Quantity world objects, which it pairs
    positionally, silently transposing the axes.
    """
    types1 = [str(t) for t in wcs1.world_axis_physical_types]
    types2 = [str(t) for t in wcs2.world_axis_physical_types]
    units1 = [unit or '' for unit in wcs1.world_axis_units]
    units2 = [unit or '' for unit in wcs2.world_axis_units]

    def convert(values, unit_in, unit_out):
        if unit_in == unit_out:
            return values
        return (np.asarray(values) * u.Unit(unit_in)).to_value(u.Unit(unit_out))

    def transform(wcs_in, wcs_out, types_in, units_in, types_out, units_out, pixel_input):
        world_in = wcs_in.pixel_to_world_values(*pixel_input)
        if wcs_in.world_n_dim == 1:
            world_in = (world_in,)
        world_out = []
        for world_axis, physical_type in enumerate(types_out):
            in_axis = types_in.index(physical_type)
            world_out.append(convert(world_in[in_axis], units_in[in_axis], units_out[world_axis]))
        return wcs_out.world_to_pixel_values(*world_out)

    def forwards(*pixel_input):
        return transform(wcs1, wcs2, types1, units1, types2, units2, pixel_input)

    def backwards(*pixel_input):
        return transform(wcs2, wcs1, types2, units2, types1, units1, pixel_input)

    return forwards, backwards


@link_helper(category='Astronomy')
class WCSLink(MultiLink):
    """
    A collection of links that link the pixel components of two datasets via
    WCS transformations.
    """

    display = 'WCS link'
    cid_independent = True

    def __init__(self, data1=None, data2=None, cids1=None, cids2=None):

        wcs1, wcs2 = data1.coords, data2.coords

        # The probe-based fast path below is only trustworthy for WCSes that
        # natively implement the high-level API: their typed world objects
        # (SkyCoord, SpectralCoord, ...) fail loudly for mismatched physical
        # types, whereas wrapped bare low-level WCSes produce plain Quantities
        # that transform positionally regardless of physical type.
        both_high_level = (isinstance(wcs1, BaseHighLevelWCS) and
                           isinstance(wcs2, BaseHighLevelWCS))

        # The high-level API is required by pixel_to_pixel below, but coords
        # may be a bare low-level (APE 14) WCS object, so wrap those.
        if not isinstance(wcs1, BaseHighLevelWCS):
            wcs1 = HighLevelWCSWrapper(wcs1)
        if not isinstance(wcs2, BaseHighLevelWCS):
            wcs2 = HighLevelWCSWrapper(wcs2)

        wcs1_ll = wcs1.low_level_wcs
        wcs2_ll = wcs2.low_level_wcs

        if (wcs1_ll.world_axis_physical_types.count(None) == wcs1_ll.world_n_dim or
            wcs2_ll.world_axis_physical_types.count(None) == wcs2_ll.world_n_dim):
            raise IncompatibleWCS(f"Can't create WCS link between {data1.label} and {data2.label}")

        if data1.ndim != wcs1_ll.pixel_n_dim or data2.ndim != wcs2_ll.pixel_n_dim:
            raise IncompatibleWCS(f"Can't create WCS link between {data1.label} and {data2.label}")

        forwards = backwards = None
        if (both_high_level and wcs1_ll.pixel_n_dim == wcs2_ll.pixel_n_dim and
                wcs1_ll.world_n_dim == wcs2_ll.world_n_dim):
            if (wcs1_ll.world_axis_physical_types.count(None) == 0 and
                    wcs2_ll.world_axis_physical_types.count(None) == 0):

                # The easiest way to check if the WCSes are compatible is to simply try and
                # see if values can be transformed for a single pixel. In future we might
                # find that this requires optimization performance-wise, but for now let's
                # not do premature optimization.

                pixel_cids1, pixel_cids2, forwards, backwards = get_cids_and_functions(wcs1, wcs2,
                                                                                       data1.pixel_component_ids[::-1],
                                                                                       data2.pixel_component_ids[::-1])

                self._physical_types_1 = wcs1_ll.world_axis_physical_types
                self._physical_types_2 = wcs2_ll.world_axis_physical_types

        if not forwards or not backwards:
            # A generalized APE 14-compatible way
            # Handle also the extra-spatial axes such as those of the time and wavelength dimensions

            # NOTE: these must not be aliased to a single list - each side's
            # axes have to be collected independently.
            wcs1_celestial_physical_types = []
            wcs2_celestial_physical_types = []

            matched_world1 = []
            matched_world2 = []

            cids1 = data1.pixel_component_ids
            cids2 = data2.pixel_component_ids

            # The celestial special case links different sky frames (e.g.
            # galactic to equatorial) through SkyCoord, and relies on
            # astropy.wcs.WCS-only attributes, so only take it for astropy WCS
            # pairs. Everything else falls through to physical-type matching.
            if (isinstance(wcs1_ll, WCS) and isinstance(wcs2_ll, WCS) and
                    wcs1_ll.has_celestial and wcs2_ll.has_celestial):
                wcs1_celestial_physical_types = wcs1_ll.celestial.world_axis_physical_types
                wcs2_celestial_physical_types = wcs2_ll.celestial.world_axis_physical_types
                matched_world1 = [wcs1_ll.wcs.lng, wcs1_ll.wcs.lat]
                matched_world2 = [wcs2_ll.wcs.lng, wcs2_ll.wcs.lat]

            wcs1_sliced_physical_types = list(wcs1_celestial_physical_types)
            wcs2_sliced_physical_types = list(wcs2_celestial_physical_types)

            for i, physical_type1 in enumerate(wcs1_ll.world_axis_physical_types):
                if physical_type1 is not None:
                    for j, physical_type2 in enumerate(wcs2_ll.world_axis_physical_types):
                        if physical_type1 == physical_type2:
                            if physical_type1 not in wcs1_sliced_physical_types:
                                matched_world1.append(i)
                                wcs1_sliced_physical_types.append(physical_type1)
                            if physical_type2 not in wcs2_sliced_physical_types:
                                matched_world2.append(j)
                                wcs2_sliced_physical_types.append(physical_type2)

            # For each matched world axis, keep the pixel axes it is
            # correlated with - APE 14 guarantees neither a one-to-one nor a
            # reversed world/pixel correspondence (e.g. celestial -TAB axes
            # couple two pixel axes to each world axis).
            slicing_axes1 = sorted(kept_numpy_axes(wcs1_ll, matched_world1), reverse=True)
            slicing_axes2 = sorted(kept_numpy_axes(wcs2_ll, matched_world2), reverse=True)

            if not slicing_axes1 or not slicing_axes2:
                raise IncompatibleWCS(f"Can't create WCS link between {data1.label} and {data2.label}")

            # Generate slices for the wcs slicing (numpy order)
            slices1 = [slice(None)] * wcs1_ll.pixel_n_dim
            slices2 = [slice(None)] * wcs2_ll.pixel_n_dim

            for i in range(wcs1_ll.pixel_n_dim):
                if i not in slicing_axes1:
                    slices1[i] = 0

            for j in range(wcs2_ll.pixel_n_dim):
                if j not in slicing_axes2:
                    slices2[j] = 0

            wcs1_sliced = SlicedLowLevelWCS(wcs1_ll, tuple(slices1))
            wcs2_sliced = SlicedLowLevelWCS(wcs2_ll, tuple(slices2))

            # slicing_axes are sorted in descending numpy-axis order, which
            # matches the pixel argument order of the sliced WCSes
            cids1_sliced = [cids1[x] for x in slicing_axes1]
            cids2_sliced = [cids2[x] for x in slicing_axes2]

            types1 = [str(t) for t in wcs1_sliced.world_axis_physical_types]
            types2 = [str(t) for t in wcs2_sliced.world_axis_physical_types]

            if not both_high_level and types1 != types2 and sorted(types1) == sorted(types2):
                if len(set(types1)) != len(types1) or 'None' in types1:
                    # Duplicated or unknown physical types cannot be paired
                    # reliably across a reordering
                    raise IncompatibleWCS(f"Can't create WCS link between {data1.label} and {data2.label}")
                # Wrapped low-level WCSes produce plain-Quantity world
                # objects, which pixel_to_pixel pairs positionally - that
                # would silently transpose the axes here, so transform
                # through explicitly type-matched world values instead.
                # Natively high-level pairs don't need this: their typed
                # world objects (e.g. SkyCoord) are matched by class.
                forwards_permuted, backwards_permuted = permuted_values_functions(wcs1_sliced, wcs2_sliced)
                pixel_cids1, pixel_cids2, forwards, backwards = get_cids_and_functions(
                    None, None, cids1_sliced, cids2_sliced,
                    forwards=forwards_permuted, backwards=backwards_permuted)
            else:
                wcs1_final = HighLevelWCSWrapper(copy.copy(wcs1_sliced))
                wcs2_final = HighLevelWCSWrapper(copy.copy(wcs2_sliced))

                pixel_cids1, pixel_cids2, forwards, backwards = get_cids_and_functions(
                    wcs1_final, wcs2_final, cids1_sliced, cids2_sliced)

            self._physical_types_1 = wcs1_sliced_physical_types
            self._physical_types_2 = wcs2_sliced_physical_types

        if pixel_cids1 is None:
            raise IncompatibleWCS(f"Can't create WCS link between {data1.label} and {data2.label}")

        super(WCSLink, self).__init__(pixel_cids1, pixel_cids2,
                                      forwards=forwards, backwards=backwards)

        self.data1 = data1
        self.data2 = data2

    def __gluestate__(self, context):
        state = {}
        state['data1'] = context.id(self.data1)
        state['data2'] = context.id(self.data2)
        return state

    @classmethod
    def __setgluestate__(cls, rec, context):
        self = cls(context.object(rec['data1']),
                   context.object(rec['data2']))
        return self

    @property
    def description(self):
        types1 = ''.join(['<li>' + phys_type for phys_type in self._physical_types_1])
        types2 = ''.join(['<li>' + phys_type for phys_type in self._physical_types_2])
        return ('This automatically links the coordinates of the '
                'two datasets using the World Coordinate System (WCS) '
                'coordinates defined in the files.<br><br>The physical types '
                'of the coordinates linked in the first dataset are: '
                f'<ul>{types1}</ul>and in the second dataset:<ul>{types2}</ul>')

    def as_affine_link(self, n_samples=1000, tolerance=1):
        """
        Approximate the link as an affine transformation which can, if the
        approximation is good, result in significant performance improvements.

        For now this will only work for datasets in which two pixel coordinates
        are linked.

        The deviation to be compared to the tolerance is measured in the frame
        of reference of the second dataset.
        """

        if len(self.cids1) != 2 or len(self.cids2) != 2:
            raise NotImplementedError("Only 2-dimensional WCS links are supported")

        # Start off by generating random positions in data1
        pixel1 = []
        for cid in self.cids1:
            size = self.data1.shape[cid.axis]
            pixel1.append(np.random.uniform(-0.5, size - 0.5, n_samples))

        # Convert to pixel positions in data2
        pixel2 = self.forwards(*pixel1)

        keep = np.ones(n_samples, dtype=bool)
        for p in pixel1 + pixel2:
            keep[np.isnan(p)] = False

        if not np.any(keep):
            raise NoAffineApproximation(f'Could not find a good affine approximation to '
                                        f'WCSLink with tolerance={tolerance}, as no overlap')

        pixel1 = [p[keep] for p in pixel1]
        pixel2 = [p[keep] for p in pixel2]

        # First try simple offset

        def transform_offset(offsets):
            pixel1_tr = pixel1[0] - offsets[0], pixel1[1] - offsets[1]
            return np.hypot(pixel2[0] - pixel1_tr[0], pixel2[1] - pixel1_tr[1])

        best, _ = leastsq(transform_offset, (0, 0))

        max_deviation = np.max(transform_offset(best))

        if max_deviation <= tolerance:
            return OffsetLink(data1=self.data1, data2=self.data2,
                              cids1=self.cids1, cids2=self.cids2, offsets=best)

        # If the above doesn't work, try a full affine transformation

        def transform_affine(coeff):
            a, b, c, d, e, f = coeff
            pixel1_tr = pixel1[0] * a + pixel1[1] * b + c, pixel1[0] * d + pixel1[1] * e + f
            return np.hypot(pixel2[0] - pixel1_tr[0], pixel2[1] - pixel1_tr[1])

        best, _ = leastsq(transform_affine, (1, 0, 0, 0, 1, 0))

        max_deviation = np.max(transform_affine(best))

        if max_deviation > tolerance:
            raise NoAffineApproximation(f'Could not find a good affine approximation to '
                                        f'WCSLink with tolerance={tolerance}')

        matrix = np.vstack([best.reshape((2, 3)), [[0, 0, 1]]])

        return AffineLink(data1=self.data1, data2=self.data2,
                          cids1=self.cids1, cids2=self.cids2, matrix=matrix)


@autolinker('Astronomy WCS')
def wcs_autolink(data_collection):

    # Find subset of datasets with WCS coordinates - low-level-only (APE 14)
    # WCS objects are accepted too, and get wrapped by WCSLink.
    wcs_datasets = [data for data in data_collection
                    if hasattr(data, 'coords') and
                    isinstance(data.coords, (BaseHighLevelWCS, BaseLowLevelWCS))]

    # Only continue if there are at least two such datasets
    if len(wcs_datasets) < 2:
        return []

    # Find existing WCS links
    existing = set()
    for link in data_collection.external_links:
        if isinstance(link, WCSLink):
            existing.add((link.data1, link.data2))

    # Loop through all pairs of datasets, skipping pairs for which a link
    # already exists. PERF: in practice we don't actually have to link all
    # pairs, so we should try and optimize that.
    all_links = []
    for i1, data1 in enumerate(wcs_datasets):
        for data2 in wcs_datasets[i1 + 1:]:
            if (data1, data2) not in existing:
                try:
                    link = WCSLink(data1, data2)
                except IncompatibleWCS:
                    continue
                all_links.append(link)

    return all_links
