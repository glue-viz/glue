import numpy as np
from glue.core.exceptions import IncompatibleDataException, IncompatibleAttribute
from glue.core.component import CoordinateComponent

__all__ = ['split_view_for_bounds', 'get_fixed_resolution_buffer']

# We should consider creating a glue-wide caching infrastructure so that we can
# better control the allowable size of the cache and centralize the cache
# invalidation. There could be a way to indicate that the cache depends on
# certain data components and/or certain subsets, so that when these are
# updated we can invalidate the cache. Although since we rely on subset states
# and not subsets, we could also just make sure that we cache based on
# subset state, and have a way to know if a subset state changes. We also
# should invalidate the cache based on links changing.


def translate_pixel(data, pixel_values, target_cid):

    if target_cid in data.pixel_component_ids:
        return pixel_values[target_cid.axis], [target_cid]

    component = data.get_component(target_cid)

    if hasattr(component, 'link'):
        link = component.link
        values = []
        cids = []
        for cid in link._from:
            v, c = translate_pixel(data, pixel_values, cid)
            values.append(v)
            cids.extend(c)
        return link._using(*values), cids
    elif isinstance(component, CoordinateComponent):
        # Hack for now - if we pass arrays in the view, it's interpreted
        return component._calculate(view=pixel_values), component.dependent_pixel_cids
    else:
        raise Exception("Dependency on non-pixel component", target_cid)


def get_fixed_resolution_buffer(data, target_data, bounds, target_cid=None, subset_state=None):

    coords = []

    pixel_values = [np.linspace(*bound) if isinstance(bound, tuple) else bound for bound in bounds]
    pixel_values = np.meshgrid(*pixel_values, indexing='ij')

    pixel_cids = []

    for ipix, pix in enumerate(data.pixel_component_ids):

        # Start off by finding all the pixel coordinates of the current
        # view in the reference frame of the current layer data.
        pixel_coord, pixs = translate_pixel(target_data, pixel_values, pix)

        pixel_cids.extend(pixs)

        original_shape = pixel_coord.shape

        coord = np.round(pixel_coord.ravel()).astype(int)

        coords.append(coord)

    pixel_cids = sorted(set([cid.axis for cid in pixel_cids]))
    for i in range(target_data.ndim):
        if isinstance(bounds[i], tuple) and i not in pixel_cids:
            raise IncompatibleDataException()

    print(coords)

    # We now do a nearest-neighbor interpolation. We don't use
    # map_coordinates because it is picky about array endian-ness
    # and if we just use normal Numpy slicing we can preserve the
    # data type (and avoid memory copies)
    keep = np.ones(len(coords[0]), dtype=bool)
    array = np.zeros(len(coords[0])) * np.nan
    for icoord, coord in enumerate(coords):
        print(icoord, coord)
        keep[(coord < 0) | (coord >= data.shape[icoord])] = False
    coords = [coord[keep] for coord in coords]

    if subset_state is None:
        array[keep] = data.get_data(target_cid, view=tuple(coords))
    else:
        array[keep] = data.get_mask(subset_state, view=tuple(coords))

    # Finally convert array back to a 2D array
    array = array.reshape(original_shape)

    slices = []
    for bound in bounds:
        if isinstance(bound, tuple):
            slices.append(slice(None))
        else:
            slices.append(0)

    array = array[slices]

    print('shape', array.shape)

    return array
