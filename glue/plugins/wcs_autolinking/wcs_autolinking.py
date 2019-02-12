from glue.config import link_wizard
from glue.core.link_helpers import multi_link
from glue.core.coordinates import WCSCoordinates
from glue.utils import efficient_pixel_to_pixel

__all__ = ['wcs_autolink']


def get_links(wcs1, wcs2, pixel_cids1, pixel_cids2):

    def forwards(*pixel_input):
        return efficient_pixel_to_pixel(wcs1, wcs2, *pixel_input)

    def backwards(*pixel_input):
        return efficient_pixel_to_pixel(wcs2, wcs1, *pixel_input)

    pixel_input = (0,) * len(pixel_cids1)

    try:
        forwards(*pixel_input)
        backwards(*pixel_input)
    except Exception:
        return None

    return multi_link(pixel_cids1, pixel_cids2,
                      forwards=forwards,
                      backwards=backwards)


def find_wcs_links(data1, data2):
    """
    Given two datasets, find pixel-world-pixel links that can be set up.
    """

    # TODO: we should avoid setting WCS on data objects where there isn't actual
    # coordinate information otherwise this function will identify that two
    # uninitialized WCSes are linked.

    # Extract WCS objects - from here onwards, we assume that these objects
    # have the new Astropy APE 14 interface.
    wcs1, wcs2 = data1.coords.wcs, data2.coords.wcs

    # Only check for links if the WCSes have well defined physical types
    if (wcs1.world_axis_physical_types.count(None) > 0 or
            wcs2.world_axis_physical_types.count(None) > 0):
        return []

    # For now, we treat the WCS as non-separable, but in future we could
    # consider iterating over the separated components of the WCS for
    # performance as well as to be able to link e.g. the celestial part of a
    # 3D WCS with a 2D WCS. So for now we require the number of pixel/world
    # coordinates to match
    if wcs1.pixel_n_dim == wcs2.pixel_n_dim and wcs1.world_n_dim == wcs2.world_n_dim:

        # The easiest way to check if the WCSes are compatible is to simply try and
        # see if values can be transformed for a single pixel. In future we might
        # find that this requires optimization performance-wise, but for now let's
        # not do premature optimization.

        link = get_links(wcs1, wcs2,
                         data1.pixel_component_ids,
                         data2.pixel_component_ids)

        if link:
            return [link]

    # Try setting only a celestial link. We try and extract the celestial
    # WCS, which will only work if the celestial coordinates are separable.
    # TODO: find a more generalized APE 14-compatible way to do this.

    if not wcs1.has_celestial or not wcs2.has_celestial:
        return []

    try:
        wcs1_celestial = wcs1.celestial
        wcs2_celestial = wcs2.celestial
    except Exception:
        return []

    cids1 = data1.pixel_component_ids
    pixel_cids1 = [cids1[wcs1.wcs.naxis - wcs1.wcs.lng - 1], cids1[wcs1.wcs.naxis - wcs1.wcs.lat - 1]]

    cids2 = data2.pixel_component_ids
    pixel_cids2 = [cids2[wcs2.wcs.naxis - wcs2.wcs.lng - 1], cids2[wcs2.wcs.naxis - wcs2.wcs.lat - 1]]

    link = get_links(wcs1_celestial, wcs2_celestial, pixel_cids1, pixel_cids2)

    if link:
        return [link]
    else:
        return []


@link_wizard('Astronomy WCS')
def wcs_autolink(data_collection):

    # Find subset of datasets with WCS coordinates
    wcs_datasets = [data for data in data_collection if isinstance(data.coords, WCSCoordinates)]

    # Only continue if there are at least two such datasets
    if len(wcs_datasets) < 2:
        return []

    # Loop through all pairs of datasets - in practice we don't actually have
    # to link all pairs, so we should try and optimize that.
    all_links = []
    for i1, data1 in enumerate(wcs_datasets):
        for data2 in wcs_datasets[i1 + 1:]:
            links = find_wcs_links(data1, data2)
            if links is not None:
                all_links.extend(links)

    return all_links
