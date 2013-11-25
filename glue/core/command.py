from abc import ABCMeta, abstractmethod

from .data_factories import load_data


class Command(object):
    """
    A class to encapsulate (and possibly undo) state changes

    Subclasses of this abstract base class must implement the
    `do` and `undo` methods.

    Both `do` and `undo` receive a single input argument named
    `context` -- this is whatever object is passed to the constructor
    of :class:`glue.core.command.CommandStack`. This object is used
    to store and retrieve resources needed by each command

    Each class should also override the kwargs class list,
    to list the keyword arguments that should be passed to the
    command instructor. The base class will check that these
    keywords are indeed provided. Commands should not take
    non-keyword arguments in the constructor method
    """
    __metaclass__ = ABCMeta
    kwargs = []

    def __init__(self, **kwargs):
        kwargs = kwargs.copy()
        for k in self.kwargs:
            if k not in kwargs:
                raise RuntimeError("Required keyword %s not passed to %s" %
                                   (k, type(self)))
            setattr(self, k, kwargs.pop(k))
        self.extra = kwargs

    @abstractmethod
    def do(self, context):
        """
        Execute the command

        :param context: An object used to store and fetch resources
                        needed by a Command.
        """
        pass

    @abstractmethod
    def undo(self, context):
        pass

class CommandStack(object):
    """The command stack collects commands,
       and saves them to enable undoing/redoing
    """
    def __init__(self, context):
        """
        :param context: An object passed to commands. Commands can
                        use this however they wish to store/fetch data
        :type context: object
        """
        self._context = context
        self._command_stack = []
        self._undo_stack = []

    def do(self, cmd):
        """
        Execute and log a new command

        :rtype: The return value of cmd.do()
        """
        self._command_stack.append(cmd)
        result = cmd.do(self._context)
        self._undo_stack = []
        return result

    def undo(self):
        """
        Undo the previous command

        :raises: IndexError, if there are no objects to undo
        """
        try:
            c = self._command_stack.pop()
        except IndexError:
            raise IndexError("No commands to undo")
        self._undo_stack.append(c)
        c.undo(self._context)

    def redo(self):
        """
        Redo the previously-undone command

        :raises: IndexError, if there are no undone actions
        """
        try:
            c = self._undo_stack.pop()
        except IndexError:
            raise IndexError("No commands to redo")
        result = c.do(self._context)
        self._command_stack.append(c)
        return result

#context needs;
# data collection
# application


# refactoring needed
# factories identifiable by unique names
class LoadData(Command):
    kwargs = ['path', 'factory']

    def do(self, context):
        return load_data(self.path, self.factory)

    def undo(self, context):
        pass


class AddData(Command):
    kwargs = ['data']

    def do(self, context):
        context.data_collection.append(self.data)

    def undo(self, context):
        context.data_collection.remove(self.data)


class RemoveData(Command):
    kwargs = ['data']

    def do(self, context):
        context.data_collection.remove(self.data)

    def undo(self, context):
        context.data_collection.append(self.data)


#refactoring needed:
# application has non-interactive new_viewer, close_viewer method
# viewer types should have a unique name
class NewDataViewer(Command):
    kwargs = ['viewer']

    def do(self, context):
        v = context.application.new_data_viewer(self.viewer)
        self.created = v
        return v

    def undo(self, context):
        self.created.close(warn=False)


class AddLayer(Command):
    kwargs = ['layer', 'viewer']

    def do(self, context):
        self.viewer.add_layer(self.layer)

    def undo(self, context):
        self.viewer.remove_layer(self.layer)


class ApplyROI(Command):
    pass


class LinkData(Command):
    pass


class SetViewState(Command):
    pass


class NewTab(Command):
    pass


class CloseTab(Command):
    pass


class NewSubset(Command):
    pass


class CopySubset(Command):
    pass


class PasteSubset(Command):
    pass


class SpecialPasteSubset(Command):
    pass


class DeleteSubset(Command):
    pass


class SetStyle(Command):
    pass


class SetLabel(Command):
    pass
