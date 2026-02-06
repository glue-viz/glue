import weakref

import numpy as np

from glue.core.data import Subset
from glue.core.exceptions import IncompatibleAttribute, IncompatibleDataException
from glue.core.link_manager import pixel_cid_to_pixel_cid_matrix


class DataProxy(object):

    def __init__(self, viewer_state, layer_artist):
        self._viewer_state = weakref.ref(viewer_state)
        self._layer_artist = weakref.ref(layer_artist)

    @property
    def layer_artist(self):
        return self._layer_artist()

    @property
    def viewer_state(self):
        return self._viewer_state()

    def _pixel_cid_order(self):
        data = self.layer_artist.layer
        if isinstance(self.layer_artist.layer, Subset):
            data = data.data
        mat = pixel_cid_to_pixel_cid_matrix(self.viewer_state.reference_data,
                                            data)
        order = []
        for i in range(mat.shape[1]):
            idx = np.argmax(mat[:, i])
            order.append(idx if mat[idx, i] else None)
        return order

    @property
    def shape(self):

        order = self._pixel_cid_order()

        try:
            x_axis = order.index(self.viewer_state.x_att.axis)
            y_axis = order.index(self.viewer_state.y_att.axis)
            z_axis = order.index(self.viewer_state.z_att.axis)
        except (AttributeError, ValueError):
            self.layer_artist.disable('Layer data is not fully linked to reference data')
            return 0, 0, 0

        if isinstance(self.layer_artist.layer, Subset):
            full_shape = self.layer_artist.layer.data.shape
        else:
            full_shape = self.layer_artist.layer.shape

        return full_shape[z_axis], full_shape[y_axis], full_shape[x_axis]

    def compute_fixed_resolution_buffer(self, bounds=None):

        shape = [bound[2] for bound in bounds]

        if self.layer_artist is None or self.viewer_state is None:
            return np.broadcast_to(0, shape)

        reference_axes = [self.viewer_state.x_att.axis,
                          self.viewer_state.y_att.axis,
                          self.viewer_state.z_att.axis]

        # For this method, we make use of Data.compute_fixed_resolution_buffer,
        # which requires us to specify bounds in the form (min, max, nsteps).
        # We also allow view to be passed here (which is a normal Numpy view)
        # and, if given, translate it to bounds. If neither are specified,
        # we behave as if view was [slice(None), slice(None), slice(None)].

        def slice_to_bound(slc, size):
            min, max, step = slc.indices(size)
            n = (max - min - 1) // step
            max = min + step * n
            return (min, max, n + 1)

        full_view, permutation = self.viewer_state.numpy_slice_permutation

        full_view[reference_axes[0]] = bounds[2]
        full_view[reference_axes[1]] = bounds[1]
        full_view[reference_axes[2]] = bounds[0]

        layer = self.layer_artist.layer
        for i in range(self.viewer_state.reference_data.ndim):
            if isinstance(full_view[i], slice):
                full_view[i] = slice_to_bound(full_view[i],
                                              self.viewer_state.reference_data.shape[i])

        if isinstance(layer, Subset):
            try:
                subset_state = layer.subset_state
                result = layer.data.compute_fixed_resolution_buffer(
                    full_view,
                    target_data=self.viewer_state.reference_data,
                    subset_state=subset_state,
                    cache_id=self.layer_artist.id)
            except (IncompatibleDataException, IncompatibleAttribute):
                self.layer_artist.disable_incompatible_subset()
                return np.broadcast_to(0, shape)
            else:
                self.layer_artist.enable()
        else:
            try:
                result = layer.compute_fixed_resolution_buffer(
                    full_view,
                    target_data=self.viewer_state.reference_data,
                    target_cid=self.layer_artist.state.attribute,
                    cache_id=self.layer_artist.id)
            except (IncompatibleDataException, IncompatibleAttribute):
                self.layer_artist.disable('Layer data is not fully linked to reference data')
                return np.broadcast_to(0, shape)
            else:
                self.layer_artist.enable()

        if permutation:
            try:
                result = result.transpose(permutation)
            except ValueError:
                return np.broadcast_to(0, shape)

        return result
