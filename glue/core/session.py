from __future__ import absolute_import, division, print_function

from glue.core.command import CommandStack
from glue.core.data_collection import DataCollection


class Session(object):

    def __init__(self, application=None, data_collection=None,
                 command_stack=None, hub=None):

        # applications can be added after instantiation
        self.application = application

        self.data_collection = data_collection or DataCollection()
        self.hub = self.data_collection.hub
        self.command_stack = command_stack or CommandStack()
        self.command_stack.session = self

        # set the global data_collection for subset updates
        from glue.core.edit_subset_mode import EditSubsetMode
        EditSubsetMode().data_collection = self.data_collection
