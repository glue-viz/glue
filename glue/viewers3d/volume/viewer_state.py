from glue.core.data import BaseData
from echo import CallbackProperty, SelectionCallbackProperty
from glue.viewers3d.common.viewer_state import ViewerState3D
from glue.core.data_combo_helper import ManualDataComboHelper

__all__ = ['VolumeViewerState3D']


class VolumeViewerState3D(ViewerState3D):

    downsample = CallbackProperty(True)
    resolution = SelectionCallbackProperty(4)
    reference_data = SelectionCallbackProperty(docstring='The dataset that is used to define the '
                                                         'available pixel/world components, and '
                                                         'which defines the coordinate frame in '
                                                         'which the images are shown')

    def __init__(self, **kwargs):

        super(VolumeViewerState3D, self).__init__()

        self.ref_data_helper = ManualDataComboHelper(self, 'reference_data')

        self.add_callback('layers', self._layers_changed)

        VolumeViewerState3D.resolution.set_choices(self, [2**i for i in range(4, 12)])

        self.update_from_dict(kwargs)

    def _first_3d_data(self):
        for layer_state in self.layers:
            if getattr(layer_state.layer, 'ndim', None) == 3:
                return layer_state.layer

    def _layers_changed(self, *args):
        self._update_combo_ref_data()
        self._set_reference_data()
        self._update_attributes()

    def _update_combo_ref_data(self, *args):
        self.ref_data_helper.set_multiple_data(self.layers_data)

    def _set_reference_data(self, *args):
        if self.reference_data is None:
            for layer in self.layers:
                if isinstance(layer.layer, BaseData):
                    self.reference_data = layer.layer
                    return

    def _update_attributes(self, *args):

        data = self._first_3d_data()

        if data is None:

            type(self).x_att.set_choices(self, [])
            type(self).y_att.set_choices(self, [])
            type(self).z_att.set_choices(self, [])

        else:

            z_cid, y_cid, x_cid = data.pixel_component_ids

            type(self).x_att.set_choices(self, [x_cid])
            type(self).y_att.set_choices(self, [y_cid])
            type(self).z_att.set_choices(self, [z_cid])

    @property
    def clip_limits_relative(self):

        data = self._first_3d_data()

        if data is None:
            return [0., 1., 0., 1., 0., 1.]
        else:
            nz, ny, nx = data.shape
            return (self.x_min / nx,
                    self.x_max / nx,
                    self.y_min / ny,
                    self.y_max / ny,
                    self.z_min / nz,
                    self.z_max / nz)
