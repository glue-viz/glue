from __future__ import absolute_import, division, print_function

import weakref

from glue.core.command import CommandStack
from glue.core.data_collection import DataCollection
from glue.core.edit_subset_mode import EditSubsetMode


class Session(object):

    def __init__(self, application=None, data_collection=None,
                 command_stack=None, hub=None):

        self.application = application
        self.data_collection = data_collection or DataCollection()
        self.hub = self.data_collection.hub
        self.command_stack = command_stack or CommandStack()
        self.command_stack.session = self

        self.edit_subset_mode = EditSubsetMode()
        self.edit_subset_mode.data_collection = self.data_collection

    @property
    def application(self):
        if self._application is None:
            return None
        else:
            return self._application()

    @application.setter
    def application(self, value):
        if value is None:
            self._application = None
        else:
            self._application = weakref.ref(value)
