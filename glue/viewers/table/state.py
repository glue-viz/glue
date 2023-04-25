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

    def __init__(self, **kwargs):

        super(TableViewerState, self).__init__()

        self.filter_att_helper = ComponentIDComboHelper(self, 'filter_att', categorical=True, numeric=False)

        self.update_from_dict(kwargs)

