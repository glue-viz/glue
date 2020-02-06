import numpy as np

from glue.core.data_derived import DerivedData

__all__ = ['PVSlicedData']


def sample_points(x, y, spacing=1):

    # Code adapted from pvextractor

    # Find the distance interval between all pairs of points
    dx = np.diff(x)
    dy = np.diff(y)
    dd = np.hypot(dx, dy)

    # Find the total displacement along the broken curve
    d = np.hstack([0., np.cumsum(dd)])

    # Figure out the number of points to sample, and stop short of the
    # last point.
    n_points = int(np.floor(d[-1] / spacing))

    if n_points == 0:
        raise ValueError("Path is shorter than spacing")

    d_sampled = np.linspace(0., n_points * spacing, n_points + 1)

    x_sampled = np.interp(d_sampled, d, x)
    y_sampled = np.interp(d_sampled, d, y)

    return x_sampled, y_sampled


class PVSlicedData(DerivedData):
    """
    A dataset where two dimensions have been replaced with one using a path.

    The extra dimension is added as the last dimension
    """

    def __init__(self, original_data, cid_x, x, cid_y, y, label=''):
        super(DerivedData, self).__init__()
        self.original_data = original_data
        self.cid_x = cid_x
        self.cid_y = cid_y
        self.set_xy(x, y)
        self.sliced_dims = (cid_x.axis, cid_y.axis)
        self._label = label

    def set_xy(self, x, y):
        x, y = sample_points(x, y)
        self.x = x
        self.y = y

    @property
    def label(self):
        return self._label

    def _without_sliced(self, iterable):
        return [x for ix, x in enumerate(iterable) if ix not in self.sliced_dims]

    @property
    def shape(self):
        return tuple(self._without_sliced(self.original_data.shape) + [len(self.x)])

    @property
    def main_components(self):
        return self.original_data.main_components

    def get_kind(self, cid):
        return self.original_data.get_kind(cid)

    def _get_pix_coords(self, view=None):

        pix_coords = []

        advanced_indexing = view is not None and isinstance(view[0], np.ndarray)

        idim_current = -1

        for idim in range(self.original_data.ndim):

            if idim == self.cid_x.axis:
                pix = self.x
                idim_current = self.ndim - 1
            elif idim == self.cid_y.axis:
                pix = self.y
                idim_current = self.ndim - 1
            else:
                pix = np.arange(self.original_data.shape[idim])
                idim_current += 1

            if view is not None and len(view) > idim_current:
                pix = pix[view[idim_current]]

            pix_coords.append(pix)

        if not advanced_indexing:
            pix_coords = np.meshgrid(*pix_coords, indexing='ij', copy=False)

        shape = pix_coords[0].shape

        keep = np.ones(shape, dtype=bool)
        for idim in range(self.original_data.ndim):
            keep &= (pix_coords[idim] >= 0) & (pix_coords[idim] < self.original_data.shape[idim])

        pix_coords = [x[keep].astype(int) for x in pix_coords]

        return pix_coords, keep, shape

    def get_data(self, cid, view=None):

        if cid in self.pixel_component_ids:
            return super().get_data(cid, view)

        pix_coords, keep, shape = self._get_pix_coords(view=view)
        result = np.zeros(shape)
        result[keep] = self.original_data.get_data(cid, view=pix_coords)

        return result

    def get_mask(self, subset_state, view=None):

        if view is None:
            view = Ellipsis

        pix_coords, keep, shape = self._get_pix_coords(view=view)
        result = np.zeros(shape)
        result[keep] = self.original_data.get_mask(subset_state, view=pix_coords)

        return result

    def compute_statistic(self, *args, view=None, **kwargs):
        pix_coords, _, _ = self._get_pix_coords(view=view)
        return self.original_data.compute_statistic(*args, view=pix_coords, **kwargs)

    def compute_histogram(self, *args, **kwargs):
        return self.original_data.compute_histogram(*args, **kwargs)

    def compute_fixed_resolution_buffer(self, *args, **kwargs):
        from glue.core.fixed_resolution_buffer import compute_fixed_resolution_buffer
        print(args, kwargs)
        return compute_fixed_resolution_buffer(self, *args, **kwargs)
