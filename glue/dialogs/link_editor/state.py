from __future__ import absolute_import, division, print_function

try:
    from inspect import getfullargspec
except ImportError:  # Python 2.7
    from inspect import getargspec as getfullargspec

# from glue.config import link_function, link_helper

from glue.core.component_link import ComponentLink
from glue.core.link_helpers import LinkCollection
from glue.core.state_objects import State
from glue.external.echo import CallbackProperty, SelectionCallbackProperty, delay_callback
from glue.core.data_combo_helper import DataCollectionComboHelper, ComponentIDComboHelper

# NOTES
# At the moment the main issue with link helpers is that they return a function
# that can just return a list of links. The issue is that in the resulting links,
# the metadata is lost. So we should probably auto-convert the output from the
# link helper into a link collection or multilink that includes metadata.

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

        if len(data_collection) == 2:
            self.data1, self.data2 = self.data_collection
        else:
            self.data1 = self.data2 = None

        self.on_data_change()

        self.add_callback('data1', self.on_data_change)
        self.add_callback('data2', self.on_data_change)

    def on_data_change(self, *args):

        if self.data1 is None or self.data2 is None:
            LinkEditorState.links.set_choices(self, [])
            return

        links = []
        for link in self._all_links:
            if ((link.data_in is self.data1 and link.data_out is self.data2) or
                    (link.data_in is self.data2 and link.data_out is self.data1)):
                links.append(link)

        with delay_callback(self, 'links'):
            LinkEditorState.links.set_choices(self, links)
            if len(links) > 0:
                self.links = links[0]

    def new_link(self, function_or_helper):

        # TODO: keep track of name of original action so that we can find it
        # again in the drop-down menu? Give IDs so that we can change the
        # display name in future?

        if hasattr(function_or_helper, 'function'):
            link = EditableLinkFunctionState(function_or_helper.function,
                                             data_in=self.data1, data_out=self.data2,
                                             output_names=function_or_helper.output_labels,
                                             description=function_or_helper.info,
                                             display=function_or_helper.function.__name__)
        else:
            link = EditableLinkFunctionState(function_or_helper.helper,
                                             data_in=self.data1, data_out=self.data2)

        self._all_links.append(link)
        with delay_callback(self, 'links'):
            self.on_data_change()
            self.links = link

    def remove_link(self):
        self._all_links.remove(self.links)
        self.on_data_change()


class EditableLinkFunctionState(State):

    function = CallbackProperty()
    data_in = CallbackProperty()
    data_out = CallbackProperty()
    description = CallbackProperty()
    display = CallbackProperty()

    def __new__(cls, function, data_in=None, data_out=None, cids_in=None,
                cid_out=None, input_names=None, output_names=None,
                display=None, description=None, helper=False):

        if isinstance(function, ComponentLink):
            input_names = function.input_names
            output_names = [function.output_name]
        elif isinstance(function, LinkCollection):
            input_names = function.labels1
            output_names = function.labels2
            description = function.description
        elif type(function) is type and issubclass(function, LinkCollection):
            input_names = function.labels1
            output_names = function.labels2
            description = function.description

        class CustomizedStateClass(EditableLinkFunctionState):
            pass

        if input_names is None:
            input_names = getfullargspec(function)[0]

        if output_names is None:
            output_names = []

        setattr(CustomizedStateClass, 'input_names', input_names)
        setattr(CustomizedStateClass, 'output_names', output_names)

        for index, input_arg in enumerate(CustomizedStateClass.input_names):
            setattr(CustomizedStateClass, input_arg, SelectionCallbackProperty(default_index=index))

        for index, output_arg in enumerate(CustomizedStateClass.output_names):
            setattr(CustomizedStateClass, output_arg, SelectionCallbackProperty(default_index=index))

        return super(EditableLinkFunctionState, cls).__new__(CustomizedStateClass)

    def __init__(self, function, data_in=None, data_out=None, cids_in=None,
                 cids_out=None, input_names=None, output_names=None,
                 display=None, description=None, helper=False):

        super(EditableLinkFunctionState, self).__init__()

        if isinstance(function, ComponentLink):
            self.function = function.get_using()
            self.inverse = function.get_inverse()
            cids_in = function.get_from_ids()
            cids_out = function.get_to_ids()
            data_in = cids_in[0].parent
            data_out = cids_out[0].parent
            self.display = self.function.__name__
            self.description = function.description
        elif isinstance(function, LinkCollection):
            self.multi_link = function
            cids_in = function.cids1
            cids_out = function.cids2
            data_in = cids_in[0].parent
            data_out = cids_out[0].parent
            self.display = function.display
            self.description = function.description
        elif type(function) is type and issubclass(function, LinkCollection):
            self.display = function.display
            self.description = function.description
        else:
            self.function = function
            self.inverse = None
            self.display = display
            self.description = description

        self.data_in = data_in
        self.data_out = data_out

        for name in self.input_names:
            helper = ComponentIDComboHelper(self, name)
            helper.append_data(data_in)
            setattr(self, '_' + name + '_helper', helper)

        for name in self.output_names:
            helper = ComponentIDComboHelper(self, name)
            helper.append_data(data_out)
            setattr(self, '_' + name + '_helper', helper)

        if cids_in is not None:
            for name, cid in zip(self.input_names, cids_in):
                setattr(self, name, cid)

        if cids_out is not None:
            for name, cid in zip(self.output_names, cids_out):
                setattr(self, name, cid)

    def __str__(self):
        return self.display

    @property
    def link(self):
        """
        Return a `glue.core.component_link.ComponentLink` object.
        """

        # FunctionalLinkCollection
        #
        # MultiLink

        if self.helper:
            raise NotImplementedError('helpers not implemented')
        else:
            cids_in = [getattr(self, name) for name in self.input_names]
            cid_out = getattr(self, self.output_names[0])
            return ComponentLink(cids_in, cid_out, using=self.function)
