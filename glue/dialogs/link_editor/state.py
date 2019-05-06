from __future__ import absolute_import, division, print_function

from types import FunctionType, MethodType

try:
    from inspect import getfullargspec
except ImportError:  # Python 2.7
    from inspect import getargspec as getfullargspec

from glue.config import link_function

from glue.core.component_link import ComponentLink
from glue.core.link_helpers import LinkCollection
from glue.core.state_objects import State
from glue.external.echo import CallbackProperty, SelectionCallbackProperty, delay_callback
from glue.core.data_combo_helper import DataCollectionComboHelper, ComponentIDComboHelper

__all__ = ['LinkEditorState', 'EditableLinkFunctionState']


class LinkEditorState(State):

    data1 = SelectionCallbackProperty()
    data2 = SelectionCallbackProperty()
    att1 = SelectionCallbackProperty()
    att2 = SelectionCallbackProperty()
    current_link = SelectionCallbackProperty()
    link_type = SelectionCallbackProperty()
    restrict_to_suggested = CallbackProperty(False)

    def __init__(self, data_collection, suggested_links=None):

        super(LinkEditorState, self).__init__()

        # Find identity function
        for func in link_function:
            if func.function.__name__ == 'identity':
                self._identity = func
                break
        else:
            raise ValueError("Could not find identity link function")

        self.data1_helper = DataCollectionComboHelper(self, 'data1', data_collection)
        self.data2_helper = DataCollectionComboHelper(self, 'data2', data_collection)

        self.att1_helper = ComponentIDComboHelper(self, 'att1', pixel_coord=True, world_coord=True)
        self.att2_helper = ComponentIDComboHelper(self, 'att2', pixel_coord=True, world_coord=True)

        # FIXME: We unregister the combo helpers straight away to avoid issues with
        # leftover references once the dialog is closed. This shouldn't happen
        # ideally so in future we should investigate how to avoid it.
        self.data1_helper.unregister(data_collection.hub)
        self.data2_helper.unregister(data_collection.hub)

        self.data_collection = data_collection

        # Convert links to editable states
        links = [EditableLinkFunctionState(link) for link in data_collection.external_links]

        # If supplied, also add suggested links and make sure we toggle the
        # suggestion flag on the link state so that we can handle suggestions
        # differently in the link viewer.
        if suggested_links is not None:
            for link in suggested_links:
                link_state = EditableLinkFunctionState(link)
                link_state.suggested = True
                links.append(link_state)

        self.links = links

        if len(data_collection) == 2:
            self.data1, self.data2 = self.data_collection
        else:
            self.data1 = self.data2 = None

        self._on_data_change()
        self._on_data1_change()
        self._on_data2_change()

        self.add_callback('data1', self._on_data1_change)
        self.add_callback('data2', self._on_data2_change)
        self.add_callback('restrict_to_suggested', self._on_data_change)

        LinkEditorState.current_link.set_display_func(self, self._display_link)

    @property
    def visible_links(self):

        if self.data1 is None or self.data2 is None:
            return []

        links = []
        for link in self.links:
            if link.suggested or not self.restrict_to_suggested:
                if ((link.data_in is self.data1 and link.data_out is self.data2) or
                        (link.data_in is self.data2 and link.data_out is self.data1)):
                    links.append(link)

        return links

    def flip_data(self, *args):
        with delay_callback(self, 'data1', 'data2'):
            self.data1, self.data2 = self.data2, self.data1

    def _on_data1_change(self, *args):
        if self.data1 is self.data2 and self.data1 is not None:
            self.data2 = next(data for data in self.data_collection if data is not self.data1)
        else:
            self._on_data_change()
        self.att1_helper.set_multiple_data([] if self.data1 is None else [self.data1])

    def _on_data2_change(self, *args):
        if self.data2 is self.data1 and self.data2 is not None:
            self.data1 = next(data for data in self.data_collection if data is not self.data2)
        else:
            self._on_data_change()
        self.att2_helper.set_multiple_data([] if self.data2 is None else [self.data2])

    def _on_data_change(self, *args):

        links = self.visible_links
        with delay_callback(self, 'current_link'):
            LinkEditorState.current_link.set_choices(self, links)
            if len(links) > 0:
                self.current_link = links[0]

    def _display_link(self, link):
        if link.suggested:
            return str(link) + ' [Suggested]'
        else:
            return str(link)

    def simple_link(self, *args):
        self.new_link(self._identity)
        self.current_link.x = self.att1
        self.current_link.y = self.att2

    def new_link(self, function_or_helper):

        if hasattr(function_or_helper, 'function'):
            link = EditableLinkFunctionState(function_or_helper.function,
                                             data_in=self.data1, data_out=self.data2,
                                             output_names=function_or_helper.output_labels,
                                             description=function_or_helper.info,
                                             display=function_or_helper.function.__name__)
        else:
            link = EditableLinkFunctionState(function_or_helper.helper,
                                             data_in=self.data1, data_out=self.data2)

        self.links.append(link)
        with delay_callback(self, 'current_link'):
            self._on_data_change()
            self.current_link = link

    def remove_link(self):
        self.links.remove(self.current_link)
        self._on_data_change()

    def update_links_in_collection(self):
        links = [link_state.link for link_state in self.links]
        self.data_collection.set_links(links)


class EditableLinkFunctionState(State):

    function = CallbackProperty()
    data_in = CallbackProperty()
    data_out = CallbackProperty()
    description = CallbackProperty()
    display = CallbackProperty()
    suggested = CallbackProperty(False)

    def __new__(cls, function, data_in=None, data_out=None, cids_in=None,
                cid_out=None, input_names=None, output_names=None,
                display=None, description=None):

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
                 display=None, description=None):

        super(EditableLinkFunctionState, self).__init__()

        if isinstance(function, ComponentLink):
            self._function = function.get_using()
            self._inverse = function.get_inverse()
            self._helper_class = None
            cids_in = function.get_from_ids()
            cids_out = function.get_to_ids()
            data_in = cids_in[0].parent
            data_out = cids_out[0].parent
            self.display = self._function.__name__
            self.description = function.description
        elif isinstance(function, LinkCollection):
            self._function = None
            self._helper_class = function.__class__
            cids_in = function.cids1
            cids_out = function.cids2
            data_in = cids_in[0].parent

            # To be backward-compatible with cases where @link_helper doesn't
            # include output labels, we need to assume cids_out can be empty
            # in which case we look for the second dataset inside cids_in
            if len(cids_out) > 0:
                data_out = cids_out[0].parent
            else:
                for cid in cids_in[1:]:
                    if cid.parent is not data_in:
                        data_out = cid.parent
                        break
                else:
                    raise ValueError("Could not determine second dataset in link")

            self.display = function.display
            self.description = function.description
            self._mode = 'helper'
        elif type(function) is type and issubclass(function, LinkCollection):
            self._function = None
            self._helper_class = function
            self.display = function.display
            self.description = function.description
        elif isinstance(function, (FunctionType, MethodType)):
            self._function = function
            self._inverse = None
            self._helper_class = None
            self.inverse = None
            self.display = display
            self.description = description
        else:
            raise TypeError("Unexpected type for 'function': {0}".format(type(function)))

        self.data_in = data_in
        self.data_out = data_out

        for name in self.input_names:
            helper = ComponentIDComboHelper(self, name,
                                            pixel_coord=True, world_coord=True)
            helper.append_data(data_in)
            helper.append_data(data_out)

            setattr(self, '_' + name + '_helper', helper)

        for name in self.output_names:
            helper = ComponentIDComboHelper(self, name,
                                            pixel_coord=True, world_coord=True)
            helper.append_data(data_out)
            helper.append_data(data_in)
            setattr(self, '_' + name + '_helper', helper)

        if cids_in is not None:
            for name, cid in zip(self.input_names, cids_in):
                setattr(self, name, cid)

        if cids_out is not None:
            for name, cid in zip(self.output_names, cids_out):
                setattr(self, name, cid)

    def __str__(self):

        if len(self.input_names) > 0 or len(self.output_names) > 0:

            # Construct display of linked cids
            cids1 = [str(getattr(self, cid)) for cid in self.input_names]
            cids2 = [str(getattr(self, cid)) for cid in self.output_names]
            cids = ','.join(cids1) + ' <-> ' + ','.join(cids2)

            return '{0}({1})'.format(self.display, cids)

        else:

            return self.display

    @property
    def link(self):
        """
        Return a `glue.core.component_link.ComponentLink` object.
        """
        if self._function is not None:
            cids_in = [getattr(self, name) for name in self.input_names]
            cid_out = getattr(self, self.output_names[0])
            return ComponentLink(cids_in, cid_out,
                                 using=self._function, inverse=self._inverse)
        else:
            cids_in = [getattr(self, name) for name in self.input_names]
            cids_out = [getattr(self, name) for name in self.output_names]
            return self._helper_class(cids1=cids_in, cids2=cids_out,
                                      data1=self.data_in, data2=self.data_out)
