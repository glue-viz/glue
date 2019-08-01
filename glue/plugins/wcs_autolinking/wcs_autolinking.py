from glue.plugins.wcs_autolinking import ASTROPY_GE_31

from glue.config import autolinker, link_helper
from glue.core.link_helpers import MultiLink
from glue.core.coordinates import WCSCoordinates
from glue.utils import efficient_pixel_to_pixel

__all__ = ['IncompatibleWCS', 'WCSLink', 'wcs_autolink']


class IncompatibleWCS(Exception):
    pass


def get_cids_and_functions(wcs1, wcs2, pixel_cids1, pixel_cids2):

    def forwards(*pixel_input):
        return efficient_pixel_to_pixel(wcs1, wcs2, *pixel_input)

    def backwards(*pixel_input):
        return efficient_pixel_to_pixel(wcs2, wcs1, *pixel_input)

    pixel_input = (0,) * len(pixel_cids1)

    try:
        forwards(*pixel_input)
        backwards(*pixel_input)
    except Exception:
        return None, None, None, None

    return pixel_cids1, pixel_cids2, forwards, backwards


@link_helper(category='Astronomy')
class WCSLink(MultiLink):
    """
    A collection of links that link the pixel components of two datasets via
    WCS transformations.
    """

    display = 'WCS link'
    cid_independent = True

    def __init__(self, data1=None, data2=None, cids1=None, cids2=None):

        # Extract WCS objects - from here onwards, we assume that these objects
        # have the new Astropy APE 14 interface.
        wcs1, wcs2 = data1.coords.wcs, data2.coords.wcs

        forwards = backwards = None
        if wcs1.pixel_n_dim == wcs2.pixel_n_dim and wcs1.world_n_dim == wcs2.world_n_dim:
            if (wcs1.world_axis_physical_types.count(None) == 0 and
                    wcs2.world_axis_physical_types.count(None) == 0):

                # The easiest way to check if the WCSes are compatible is to simply try and
                # see if values can be transformed for a single pixel. In future we might
                # find that this requires optimization performance-wise, but for now let's
                # not do premature optimization.

                pixel_cids1, pixel_cids2, forwards, backwards = get_cids_and_functions(wcs1, wcs2,
                                                                                       data1.pixel_component_ids,
                                                                                       data2.pixel_component_ids)

                self._physical_types_1 = wcs1.world_axis_physical_types
                self._physical_types_2 = wcs2.world_axis_physical_types

        if not forwards or not backwards:
            # Try setting only a celestial link. We try and extract the celestial
            # WCS, which will only work if the celestial coordinates are separable.
            # TODO: find a more generalized APE 14-compatible way to do this.

            if not wcs1.has_celestial or not wcs2.has_celestial:
                raise IncompatibleWCS("Can't create WCS link between {0} and {1}".format(data1.label, data2.label))

            try:
                wcs1_celestial = wcs1.celestial
                wcs2_celestial = wcs2.celestial
            except Exception:
                raise IncompatibleWCS("Can't create WCS link between {0} and {1}".format(data1.label, data2.label))

            cids1 = data1.pixel_component_ids
            cids1_celestial = [cids1[wcs1.wcs.naxis - wcs1.wcs.lng - 1],
                                     cids1[wcs1.wcs.naxis - wcs1.wcs.lat - 1]]

            cids2 = data2.pixel_component_ids
            cids2_celestial = [cids2[wcs2.wcs.naxis - wcs2.wcs.lng - 1],
                                     cids2[wcs2.wcs.naxis - wcs2.wcs.lat - 1]]

            pixel_cids1, pixel_cids2, forwards, backwards = get_cids_and_functions(wcs1_celestial, wcs2_celestial,
                                                                                   cids1_celestial, cids2_celestial)

            self._physical_types_1 = wcs1_celestial.world_axis_physical_types
            self._physical_types_2 = wcs2_celestial.world_axis_physical_types

        if pixel_cids1 is None:
            raise IncompatibleWCS("Can't create WCS link between {0} and {1}".format(data1.label, data2.label))

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
        if not ASTROPY_GE_31:
            raise ValueError("Loading this session file requires Astropy 3.1 "
                             "or later to be installed")
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
                '<ul>{0}</ul>and in the second dataset:<ul>{1}</ul>'
                .format(types1, types2))


@autolinker('Astronomy WCS')
def wcs_autolink(data_collection):

    # Find subset of datasets with WCS coordinates
    wcs_datasets = [data for data in data_collection if isinstance(data.coords, WCSCoordinates)]

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
