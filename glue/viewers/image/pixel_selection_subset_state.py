import numpy as np

from glue.core.subset import SliceSubsetState
from glue.core.exceptions import IncompatibleAttribute
from glue.core.link_manager import pixel_cid_to_pixel_cid_matrix

__all__ = ['PixelSubsetState']


class PixelSubsetState(SliceSubsetState):

    def copy(self):
        return PixelSubsetState(self.reference_data, self.slices)

    def to_array(self, data, att):

        try:

            return super(PixelSubsetState, self).to_array(data, att)

        except IncompatibleAttribute:

            if data is not self.reference_data:
                pix_coord_out = self._to_linked_pixel_coords(data)
                pix_coord_out = tuple([slice(None) if p is None else slice(p, p + 1) for p in pix_coord_out])
                return data[att, pix_coord_out]

        raise IncompatibleAttribute()

    def _to_linked_pixel_coords(self, data):

        # Determine which pixel dimensions are being sliced over
        dimensions = [idim for idim, slc in enumerate(self.slices) if slc.start is not None]

        # Determine pixel to pixel correlation matrix
        matrix = pixel_cid_to_pixel_cid_matrix(self.reference_data, data)

        # Find pixel dimensions in 'data' that are correlated
        correlated_dims = np.nonzero(np.any(matrix[dimensions], axis=0))[0]

        # Check that if we do the operation backwards we just get the
        # original dimensions back
        check_dimensions = np.nonzero(np.any(matrix[:, correlated_dims], axis=1))[0]

        if np.array_equal(dimensions, check_dimensions):
            pix_coord_in = tuple([slice(0, 1) if slc.start is None else slc for slc in self.slices])
            pix_coord_out = []
            for idim, pix_cid in enumerate(data.pixel_component_ids):
                if idim in correlated_dims:
                    coord = int(np.round(self.reference_data[pix_cid, pix_coord_in].ravel()[0]))
                else:
                    coord = None
                pix_coord_out.append(coord)

            return pix_coord_out

        raise IncompatibleAttribute()

    def get_xy(self, data, dim1, dim2):
        pix_coord_out = self._to_linked_pixel_coords(data)
        if pix_coord_out[dim1] is None or pix_coord_out[dim2] is None:
            raise IncompatibleAttribute
        else:
            return pix_coord_out[dim1], pix_coord_out[dim2]
