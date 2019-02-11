from glue.config import link_wizard
from glue.core.link_helpers import multi_link
from glue.core.coordinates import WCSCoordinates
from glue.utils import pixel_to_pixel_wrapper

__all__ = ['wcs_autolink']


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

    # For now, we treat the WCS as non-separable, but in future we could
    # consider iterating over the separated components of the WCS for
    # performance as well as to be able to link e.g. the celestial part of a
    # 3D WCS with a 2D WCS. So for now we require the number of pixel/world
    # coordinates to match
    if wcs1.pixel_n_dim != wcs2.pixel_n_dim or wcs1.world_n_dim != wcs2.world_n_dim:
        return []

    # The easiest way to check if the WCSes are compatible is to simply try and
    # see if values can be transformed for a single pixel. In future we might
    # find that this requires optimization performance-wise, but for now let's
    # not do premature optimization.

    def forwards(*pixel_input):
        return pixel_to_pixel_wrapper(wcs1, wcs2, *pixel_input)

    def backwards(*pixel_input):
        return pixel_to_pixel_wrapper(wcs2, wcs1, *pixel_input)

    pixel_input = (0,) * wcs1.pixel_n_dim
    try:
        forwards(*pixel_input)
        backwards(*pixel_input)
    except Exception:
        return []

    # If we get here, the two WCSes are compatible and we can set up a link

    link = multi_link(data1.pixel_component_ids,
                      data2.pixel_component_ids,
                      forwards=forwards,
                      backwards=backwards)

    return [link]


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
