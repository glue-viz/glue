import numpy as np
from glue.core.exceptions import IncompatibleDataException, IncompatibleAttribute
from glue.core.component import CoordinateComponent

__all__ = ['get_fixed_resolution_buffer']

# We should consider creating a glue-wide caching infrastructure so that we can
# better control the allowable size of the cache and centralize the cache
# invalidation. There could be a way to indicate that the cache depends on
# certain data components and/or certain subsets, so that when these are
# updated we can invalidate the cache. Although since we rely on subset states
# and not subsets, we could also just make sure that we cache based on
# subset state, and have a way to know if a subset state changes. We also
# should invalidate the cache based on links changing.


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


def get_fixed_resolution_buffer(data, bounds, target_data=None, target_cid=None, subset_state=None, broadcast=True):
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

    # Start off by generating arrays of coordinates in the original dataset
    pixel_coords = [np.linspace(*bound) if isinstance(bound, tuple) else bound for bound in bounds]
    pixel_coords = np.meshgrid(*pixel_coords, indexing='ij')

    # Keep track of the original shape of these arrays
    original_shape = pixel_coords[0].shape

    # Now loop through the dimensions of 'data' to find the corresponding
    # coordinates in the frame of view of this dataset.

    translated_coords = []
    dimensions_all = []

    for ipix, pix in enumerate(data.pixel_component_ids):
        translated_coord, dimensions = translate_pixel(target_data, pixel_coords, pix)
        translated_coords.append(np.round(translated_coord.ravel()).astype(int))
        dimensions_all.extend(dimensions)

    if data is not target_data and not broadcast:
        for i in range(target_data.ndim):
            if isinstance(bounds[i], tuple) and i not in dimensions_all:
                raise IncompatibleDataException()

    # We now do a nearest-neighbor interpolation. We don't use
    # map_coordinates because it is picky about array endian-ness
    # and if we just use normal Numpy slicing we can preserve the
    # data type (and avoid memory copies)
    keep = np.ones(len(translated_coords[0]), dtype=bool)
    array = np.zeros(len(translated_coords[0])) * np.nan
    for icoord, coord in enumerate(translated_coords):
        keep[(coord < 0) | (coord >= data.shape[icoord])] = False
    coords = [coord[keep] for coord in translated_coords]

    # Take subset_state into account, if present
    if subset_state is None:
        array[keep] = data.get_data(target_cid, view=tuple(coords))
    else:
        array[keep] = data.get_mask(subset_state, view=tuple(coords))

    # Finally convert array back to an n-D array
    array = array.reshape(original_shape)

    # Drop dimensions for which bounds were scalars
    slices = []
    for bound in bounds:
        if isinstance(bound, tuple):
            slices.append(slice(None))
        else:
            slices.append(0)

    return array[slices]
