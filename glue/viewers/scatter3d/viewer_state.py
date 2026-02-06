from glue.core.data_combo_helper import ComponentIDComboHelper
from glue.viewers.common3d.viewer_state import ViewerState3D

__all__ = ['ScatterViewerState3D']


class ScatterViewerState3D(ViewerState3D):

    def __init__(self, **kwargs):

        super(ScatterViewerState3D, self).__init__()

        self.x_att_helper = ComponentIDComboHelper(self, 'x_att', categorical=False)
        self.y_att_helper = ComponentIDComboHelper(self, 'y_att', categorical=False)
        self.z_att_helper = ComponentIDComboHelper(self, 'z_att', categorical=False)

        self.add_callback('layers', self._on_layers_change)

        self.update_from_dict(kwargs)

    def _on_layers_change(self, *args):
        layers_data = [layer_state.layer for layer_state in self.layers]
        self.x_att_helper.set_multiple_data(layers_data)
        self.y_att_helper.set_multiple_data(layers_data)
        self.z_att_helper.set_multiple_data(layers_data)
