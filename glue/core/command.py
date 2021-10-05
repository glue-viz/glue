import weakref
import logging
from abc import ABCMeta, abstractmethod

from glue.utils import CallbackMixin
from glue.core.data_factories import load_data
from glue.core.edit_subset_mode import ReplaceMode

MAX_UNDO = 50
"""
The classes in this module allow user actions to be stored as commands,
which can be undone/redone

All UI frontends should map interactions to command objects, instead
of directly performing an action.

Commands have access to two sources of data: the first are the
keyword arguments passed to the constructor. These are stored as
attributes of self. The second is a session object passed to all
Command.do and Command.undo calls.
"""


class Command(object):

    """
    A class to encapsulate (and possibly undo) state changes

    Subclasses of this abstract base class must implement the
    `do` and `undo` methods.

    Both `do` and `undo` receive a single input argument named
    `session` -- this is whatever object is passed to the constructor
    of :class:`glue.core.command.CommandStack`. This object is used
    to store and retrieve resources needed by each command. The
    Glue application itself uses a :class:`~glue.core.session.Session`
    instance for this.

    Each class should also override the class-level kwargs list,
    to list the required keyword arguments that should be passed to the
    command constructor. The base class will check that these
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
    def do(self, session):
        """
        Execute the command.

        Parameters
        ----------
        session : object
            An object used to store and fetch resources needed by `Command`.
        """
        pass

    @abstractmethod
    def undo(self, session):
        pass

    @property
    def label(self):
        return type(self).__name__


class CommandStack(CallbackMixin):

    """
    The command stack collects commands,
    and saves them to enable undoing/redoing

    After instantiation, something can be assigned to
    the session property. This is passed as the sole argument
    of all Command (un)do methods.
    """

    def __init__(self):
        super(CommandStack, self).__init__()
        self._session = None
        self._command_stack = []
        self._undo_stack = []

    @property
    def session(self):
        return self._session

    @session.setter
    def session(self, value):
        self._session = value

    @property
    def undo_label(self):
        """ Brief label for the command reversed by an undo """
        if len(self._command_stack) == 0:
            return ''
        cmd = self._command_stack[-1]
        return cmd.label

    @property
    def redo_label(self):
        """ Brief label for the command executed on a redo"""
        if len(self._undo_stack) == 0:
            return ''
        cmd = self._undo_stack[-1]
        return cmd.label

    def do(self, cmd):
        """
        Execute and log a new command.

        Parameters
        ----------
        cmd : :class:`Command`

        Returns
        -------
        result
            The return value of ``cmd.do()``.
        """
        logging.getLogger(__name__).debug("Do %s", cmd)
        self._command_stack.append(cmd)
        result = cmd.do(self._session)
        self._command_stack = self._command_stack[-MAX_UNDO:]
        self._undo_stack = []
        self.notify('do')
        return result

    def undo(self):
        """
        Undo the previous command.

        Raises
        ------
        IndexError, if there are no objects in the stack to undo.
        """
        try:
            c = self._command_stack.pop()
            logging.getLogger(__name__).debug("Undo %s", c)
        except IndexError:
            raise IndexError("No commands to undo")
        self._undo_stack.append(c)
        c.undo(self._session)
        self.notify('undo')

    def redo(self):
        """
        Redo the previously-undone command.

        Raises
        ------
        IndexError, if there are no undone actions.
        """
        try:
            c = self._undo_stack.pop()
            logging.getLogger(__name__).debug("Undo %s", c)
        except IndexError:
            raise IndexError("No commands to redo")
        result = c.do(self._session)
        self._command_stack.append(c)
        self.notify('redo')
        return result

    def can_undo_redo(self):
        """
        Return whether undo and redo options are possible.

        Returns
        -------
        tuple(bool, bool)
            Whether undo and redo are possible, respectively.
        """
        return len(self._command_stack) > 0, len(self._undo_stack) > 0


class LoadData(Command):
    kwargs = ['path', 'factory']
    label = 'load data'

    def do(self, session):
        return load_data(self.path, self.factory)

    def undo(self, session):
        pass


class AddData(Command):
    kwargs = ['data']
    label = 'add data'

    def do(self, session):
        session.data_collection.append(self.data)

    def undo(self, session):
        session.data_collection.remove(self.data)


class RemoveData(Command):
    kwargs = ['data']
    label = 'remove data'

    def do(self, session):
        session.data_collection.remove(self.data)

    def undo(self, session):
        session.data_collection.append(self.data)


class NewDataViewer(Command):
    """
    Add a new data viewer to the application.

    Parameters
    ----------
    viewer : `type`
        The class of viewer to create.
    data : :class:`~glue.core.BaseData` or :class:`~glue.core.subset.Subset`, optional
        The data object to initialize the viewer with.
    """
    kwargs = ['viewer', 'data']
    label = 'new data viewer'

    def do(self, session):
        viewer = session.application.new_data_viewer(self.viewer, self.data)
        if viewer is not None:
            self.created = weakref.ref(viewer)
        return viewer

    def undo(self, session):
        created = self.created()
        if created is not None:
            created.close(warn=False)


class AddLayer(Command):
    """
    Add a new layer to a viewer.

    Parameters
    ----------
    layer : :class:`~glue.core.data.Data` or :class:`~glue.core.subset.Subset`
        The layer to add.
    viewer : :class:`~glue.viewers.common.viewer.BaseViewer`
        The viewer to add the layer to.
    """
    kwargs = ['layer', 'viewer']
    label = 'add layer'

    def do(self, session):
        self.viewer.add_layer(self.layer)

    def undo(self, session):
        self.viewer.remove_layer(self.layer)


class ApplyROI(Command):
    """
    Apply an ROI to a data collection, updating subset states

    Parameters
    ----------
    data_collection: :class:`~glue.core.data_collection.DataCollection`
        DataCollection to operate on
    roi: :class:`~glue.core.roi.Roi`
        ROI to apply
    apply_func: `callable`
        The function to call which takes the ROI and actually applies it.
    """
    kwargs = ['data_collection', 'roi', 'apply_func']
    label = 'apply ROI'

    def do(self, session):
        self.old_states = {}
        for data in self.data_collection:
            for subset in data.subsets:
                self.old_states[subset] = subset.subset_state

        self.apply_func(self.roi)

    def undo(self, session):
        for data in self.data_collection:
            for subset in data.subsets:
                if subset not in self.old_states:
                    subset.delete()

        for k, v in self.old_states.items():
            k.subset_state = v


class ApplySubsetState(Command):
    """
    Apply an ROI to a data collection, updating subset states

    Parameters
    ----------
    data_collection: :class:`~glue.core.data_collection.DataCollection`
        DataCollection to operate on
    subset_state: :class:`~glue.core.subset_state.SubsetState`
        Subset state to apply
    override_mode: `bool`
        Flag indicating whether to update current subset or create a new one
    """
    kwargs = ['data_collection', 'subset_state']
    label = 'apply subset'

    def do(self, session):

        self.old_states = {}
        for data in self.data_collection:
            for subset in data.subsets:
                self.old_states[subset] = subset.subset_state

        mode = session.edit_subset_mode
        override_mode = self.extra.get('override_mode')
        # when creating a new subset the creation mode is replace mode
        # fix bug #1931
        if override_mode is None:
            if len(mode._edit_subset) == 0:
                override_mode = ReplaceMode

        mode.update(self.data_collection, self.subset_state, override_mode=override_mode)

    def undo(self, session):
        for data in self.data_collection:
            for subset in data.subsets:
                if subset not in self.old_states:
                    subset.delete()

        for k, v in self.old_states.items():
            k.subset_state = v


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
