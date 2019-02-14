from __future__ import absolute_import, division, print_function

try:
    from inspect import getfullargspec
except ImportError:  # Python 2.7
    from inspect import getargspec as getfullargspec

from glue.config import link_function, link_helper

from glue.core.state_objects import State
from glue.external.echo import CallbackProperty, SelectionCallbackProperty
from glue.core.data_combo_helper import DataCollectionComboHelper, ComponentIDComboHelper


__all__ = ['LinkEditorState']


class LinkWrapper(State):
    link = CallbackProperty()


class LinkEditorState(State):

    data1 = SelectionCallbackProperty()
    data2 = SelectionCallbackProperty()
    links = SelectionCallbackProperty()
    link_type = SelectionCallbackProperty()

    def __init__(self, data_collection, links):

        # Note that we could access the links in the data collection, but we
        # instead want to use the edited list of links in the links variable

        super(LinkEditorState, self).__init__()

        self.data1_helper = DataCollectionComboHelper(self, 'data1', data_collection)
        self.data2_helper = DataCollectionComboHelper(self, 'data2', data_collection)

        self.data_collection = data_collection
        self._all_links = links

        self.add_callback('data1', self.on_data_change)
        self.add_callback('data2', self.on_data_change)

    def on_data_change(self, *args):

        if self.data1 is None or self.data2 is None:
            LinkEditorState.links.set_choices(self, [])
            return

        links = []
        for link in self._all_links:
            to_ids = link.get_to_ids()
            to_data = [to_id.parent for to_id in to_ids]
            from_data = [from_id.parent for from_id in link.get_from_ids()]
            if ((self.data1 in to_data and self.data2 in from_data) or
                    (self.data1 in from_data and self.data2 in to_data)):
                links.append(LinkWrapper(link=link))

        LinkEditorState.links.set_choices(self, links)

    def add_link(self, function_or_helper):

        if hasattr(function_or_helper, 'function'):
            link = EditableLinkFunctionState(function_or_helper.function, self.data1, self.data2)
            print(dir(link))
        else:
            raise NotImplementedError("link helper support not implemented yet")


class EditableLinkFunctionState(State):

    # TODO: At the moment if we just wrap the inner function we lose information
    # about the 'info' and 'output_label'. So for now we just hard code 'output'
    # below.

    function = CallbackProperty()

    def __new__(cls, function, data_in, data_out, cids_in=None, cid_out=None):

        cls._input_names = getfullargspec(function)[0]
        cls._output_name = 'output'

        for index, input_arg in enumerate(cls._input_names):
            setattr(cls, input_arg, SelectionCallbackProperty(default_index=index))

        setattr(cls, cls._output_name, SelectionCallbackProperty(default_index=0))

        return super(EditableLinkFunctionState, cls).__new__(cls)

    def __init__(self, function, data_in, data_out, cids_in=None, cid_out=None):
        super(EditableLinkFunctionState, self).__init__()

        self.function = function

        for name in self._input_names:
            helper = ComponentIDComboHelper(self, name)
            helper.append_data(data_in)
            setattr(self, '_' + name + '_helper', helper)

        helper = ComponentIDComboHelper(self, self._output_name)
        setattr(self, '_' + self._output_name + '_helper', helper)
        helper.append_data(data_out)

        if cids_in is not None:
            for name, cid in zip(self._input_names, cids_in):
                setattr(self, name, cid)

        if cid_out is not None:
            setattr(self, self._output_name, cid_out)
