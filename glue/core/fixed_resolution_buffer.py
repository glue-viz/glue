import numpy as np
from glue.core.exceptions import IncompatibleDataException
from glue.core.component import CoordinateComponent
from glue.utils import unbroadcast, broadcast_to

# TODO: cache needs to be updated when links are removed/changed

__all__ = ['compute_fixed_resolution_buffer']


ARRAY_CACHE = {}
PIXEL_CACHE = {}


def translate_pixel(data, pixel_coords, target_cid):
    """
    Given a dataset and pixel coordinates in that dataset, compute a
    corresponding pixel coordinate for another dataset.

    Parameters
    ----------
    data : `~glue.core.data.Data`
        The main data object in which the original pixel coordinates are defined.
    pixel_coords : tuple or list
        List of pixel coordinates in the original data object. This should contain
        as many Numpy arrays as there are dimensions in ``data``.
    target_cid : `~glue.core.component_id.ComponentID`
        The component to compute - this can be for a different dataset.

    Returns
    -------
    coords : `~numpy.ndarray`
        The values of the translated coordinates
    dimensions : tuple
        The dimensions in ``data`` for which pixel coordinates were used in the
        translation.
    """

    if not len(pixel_coords) == data.ndim:
        raise ValueError('The number of coordinates in pixel_coords does not '
                         'match the number of dimensions in data')

    if target_cid in data.pixel_component_ids:
        return pixel_coords[target_cid.axis], [target_cid.axis]

    # TODO: check that things are efficient if the PixelComponentID is in a
    # pixel-aligned dataset.

    component = data.get_component(target_cid)

    if hasattr(component, 'link'):
        link = component.link
        values_all = []
        dimensions_all = []
        for cid in link._from:
            values, dimensions = translate_pixel(data, pixel_coords, cid)
            values_all.append(values)
            dimensions_all.extend(dimensions)
        return link._using(*values_all), dimensions_all
    elif isinstance(component, CoordinateComponent):
        # FIXME: Hack for now - if we pass arrays in the view, it's interpreted
        return component._calculate(view=pixel_coords), data.coords.dependent_axes(component.axis)
    else:
        raise Exception("Dependency on non-pixel component", target_cid)


class AnyScalar(object):
    def __eq__(self, other):
        return np.isscalar(other)


def bounds_for_cache(bounds, dimensions):
    cache_bounds = []
    for i in range(len(bounds)):
        if i not in dimensions and np.isscalar(bounds[i]):
            cache_bounds.append(AnyScalar())
        else:
            cache_bounds.append(bounds[i])
    return cache_bounds


def compute_fixed_resolution_buffer(data, bounds, target_data=None, target_cid=None,
                                    subset_state=None, broadcast=True, cache_id=None):
    """
    Get a fixed-resolution buffer for a dataset.

    Parameters
    ----------
    data : `~glue.core.Data`
        The dataset from which to extract a fixed resolution buffer
    bounds : list
        The list of bounds for the fixed resolution buffer. This list should
        have as many items as there are dimensions in ``target_data``. Each
        item should either be a scalar value, or a tuple of ``(min, max, nsteps)``.
    target_data : `~glue.core.Data`, optional
        The data in whose frame of reference the bounds are defined. Defaults
        to ``data``.
    target_cid : `~glue.core.component_id.ComponentID`, optional
        If specified, gives the component ID giving the component to use for the
        data values. Alternatively, use ``subset_state`` to get a subset mask.
    subset_state : `~glue.core.subset.SubsetState`, optional
        If specified, gives the subset state for which to compute a mask.
        Alternatively, use ``target_cid`` if you want to get data values.
    broadcast : bool, optional
        If `True`, then if a dimension in ``target_data`` for which ``bounds``
        is not a scalar does not affect any of the dimensions in ``data``,
        then the final array will be effectively broadcast along this
        dimension, otherwise an error will be raised.
    """

    if target_data is None:
        target_data = data

    if target_cid is None and subset_state is None:
        raise ValueError("Either target_cid or subset_state should be specified")

    if target_cid is not None and subset_state is not None:
        raise ValueError("Either target_cid or subset_state should be specified (not both)")

    for bound in bounds:
        if isinstance(bound, tuple) and bound[2] < 1:
            raise ValueError("Number of steps in bounds should be >=1")

    # If cache_id is specified, we keep a cached version of the resulting array
    # indexed by cache_id as well as a hash formed of the call arguments to this
    # function. We then check if the resulting array already exists in the cache.

    if cache_id is not None:

        if subset_state is None:
            # Use uuid for component ID since otherwise component IDs don't return
            # False when comparing two different CIDs (instead they return a subset state).
            # For bounds we use a special wrapper that can identify wildcards.
            current_array_hash = (data, bounds, target_data, target_cid.uuid, broadcast)
        else:
            current_array_hash = (data, bounds, target_data, subset_state, broadcast)

        current_pixel_hash = (data, target_data)

        if cache_id in ARRAY_CACHE:
            if ARRAY_CACHE[cache_id]['hash'] == current_array_hash:
                return ARRAY_CACHE[cache_id]['array']

        # To save time later, if the pixel cache doesn't match at the level of the
        # data and target_data, we just reset the cache.
        if cache_id in PIXEL_CACHE:
            if PIXEL_CACHE[cache_id]['hash'] != current_pixel_hash:
                PIXEL_CACHE.pop(cache_id)

    # Start off by generating arrays of coordinates in the original dataset
    pixel_coords = [np.linspace(*bound) if isinstance(bound, tuple) else bound for bound in bounds]
    pixel_coords = np.meshgrid(*pixel_coords, indexing='ij', copy=False)

    # Keep track of the original shape of these arrays
    original_shape = pixel_coords[0].shape

    # Now loop through the dimensions of 'data' to find the corresponding
    # coordinates in the frame of view of this dataset.

    translated_coords = []
    dimensions_all = []

    invalid_all = np.zeros(original_shape, dtype=bool)

    for ipix, pix in enumerate(data.pixel_component_ids):

        # At this point, if cache_id is in PIXEL_CACHE, we know that data and
        # target_data match so we just check the bounds. Note that the bounds
        # include the AnyScalar wildcard for any dimensions that don't impact
        # the pixel coordinates here. We do this so that we don't have to
        # recompute the pixel coordinates when e.g. slicing through cubes.

        if cache_id in PIXEL_CACHE and ipix in PIXEL_CACHE[cache_id] and PIXEL_CACHE[cache_id][ipix]['bounds'] == bounds:

            translated_coord = PIXEL_CACHE[cache_id][ipix]['translated_coord']
            dimensions = PIXEL_CACHE[cache_id][ipix]['dimensions']
            invalid = PIXEL_CACHE[cache_id][ipix]['invalid']

        else:

            translated_coord, dimensions = translate_pixel(target_data, pixel_coords, pix)

            # The returned coordinates may often be a broadcasted array. To convert
            # the coordinates to integers and check which ones are within bounds, we
            # thus operate on the un-broadcasted array, before broadcasting it back
            # to the original shape.
            translated_coord = np.round(unbroadcast(translated_coord)).astype(int)
            invalid = (translated_coord < 0) | (translated_coord >= data.shape[ipix])

            # Since we are going to be using these coordinates later on to index an
            # array, we need the coordinates to be within the array, so we reset
            # any invalid coordinates and keep track of which pixels are invalid
            # to reset them later.
            translated_coord[invalid] = 0

            # We now populate the cache
            if cache_id is not None:

                if cache_id not in PIXEL_CACHE:
                    PIXEL_CACHE[cache_id] = {'hash': current_pixel_hash}

                PIXEL_CACHE[cache_id][ipix] = {'translated_coord': translated_coord,
                                               'dimensions': dimensions,
                                               'invalid': invalid,
                                               'bounds': bounds_for_cache(bounds, dimensions)}

        invalid_all |= invalid

        # Broadcast back to the original shape and add to the list
        translated_coords.append(broadcast_to(translated_coord, original_shape))

        # Also keep track of all the dimensions that contributed to this coordinate
        dimensions_all.extend(dimensions)

    # If a dimension from the target data for which bounds was set to an interval
    # did not actually contribute to any of the coordinates in data, then if
    # broadcast is set to False we raise an error, otherwise we proceed and
    # implicitly broadcast values along that dimension of the target data.

    if data is not target_data and not broadcast:
        for i in range(target_data.ndim):
            if isinstance(bounds[i], tuple) and i not in dimensions_all:
                raise IncompatibleDataException()

    # PERF: optimize further - check if we can extract a sub-region that
    # contains all the valid values.

    # We should avoid accessing the data using tuples of 1D arrays if possible
    # since that can be very inefficient. Instead, we should pre-fetch a regular
    # array and then use array indexing on that smaller array. This should help
    # for cases where e.g. the array is memory mapped or using dask.

    preliminary_view = []
    for idim in range(len(translated_coords)):
        imin = np.min(translated_coords[idim])
        imax = np.max(translated_coords[idim])
        preliminary_view.append(slice(imin, imax + 1))
        translated_coords[idim] = translated_coords[idim] - imin
    preliminary_view = tuple(preliminary_view)

    translated_coords = tuple(translated_coords)

    print("Optimizing FRB by getting preliminary view:", preliminary_view)

    # Take subset_state into account, if present
    if subset_state is None:
        array = data.get_data(target_cid, view=preliminary_view).astype(float)
        invalid_value = -np.inf
    else:
        array = data.get_mask(subset_state, view=preliminary_view)
        invalid_value = False

    # Now apply the array indices
    array = array[translated_coords]

    if np.any(invalid_all):
        if not array.flags.writeable:
            array = np.array(array, dtype=type(invalid_value))
        array[invalid_all] = invalid_value

    # Drop dimensions for which bounds were scalars
    slices = []
    for bound in bounds:
        if isinstance(bound, tuple):
            slices.append(slice(None))
        else:
            slices.append(0)

    array = array[tuple(slices)]

    if cache_id is not None:

        # For the bounds, we use a special wildcard for bounds that don't affect
        # the result. This will allow the cache to match regardless of the
        # value for those bounds. However, we only do this for scalar bounds.

        cache_bounds = bounds_for_cache(bounds, dimensions_all)

        current_array_hash = current_array_hash[:1] + (cache_bounds,) + current_array_hash[2:]

        if subset_state is None:
            ARRAY_CACHE[cache_id] = {'hash': current_array_hash, 'array': array}
        else:
            ARRAY_CACHE[cache_id] = {'hash': current_array_hash, 'array': array}

    return array
