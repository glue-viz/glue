from echo import CallbackProperty, SelectionCallbackProperty
from glue.core.data_combo_helper import ComponentIDComboHelper

from glue.viewers.common.state import ViewerState


__all__ = ['TableViewerState']


class TableViewerState(ViewerState):
    """
    A state class that includes all the attributes for a table viewer.
    """

    filter_att = SelectionCallbackProperty(docstring='The component/column to filter/search on', default_index=0)
    filter = CallbackProperty(docstring='The text string to filter/search on')
    regex = CallbackProperty(docstring='Whether to apply regex to filter/search', default=False)

    def __init__(self, **kwargs):

        super(TableViewerState, self).__init__()
        self.filter_att_helper = ComponentIDComboHelper(self, 'filter_att', categorical=True, numeric=False)
        self.add_callback('layers', self._layers_changed)

        self.update_from_dict(kwargs)

    def _layers_changed(self, *args):

        layers_data = self.layers_data

        layers_data_cache = getattr(self, '_layers_data_cache', [])

        if layers_data == layers_data_cache:
            return

        self.filter_att_helper.set_multiple_data(self.layers_data)

        self._layers_data_cache = layers_data

    def _update_priority(self, name):
        if name == 'layers':
            return 2
        else:
            return 1
