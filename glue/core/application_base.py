from functools import wraps
import traceback

from .hub import HubListener, Hub
from .data_collection import DataCollection
from .data_factories import load_data
from .command import CommandStack
from .session import Session


def catch_error(msg):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                m = "%s\n%s" % (msg, e.message)
                detail = str(traceback.format_exc())
                self = args[0]
                self.report_error(m, detail)
        return wrapper
    return decorator


class Application(HubListener):

    def __init__(self, data_collection=None, hub=None):

        self._data = data_collection or DataCollection()
        self._hub = hub or Hub(self._data)
        self._cmds = CommandStack()
        session = Session(data_collection=self._data,
                          application=self, command_stack=self._cmds,
                          hub=self._hub)
        self._cmds.session = session
        self._session = session


    def new_data_viewer(self, viewer_class, data=None):
        """
        Create a new data viewer, add it to the UI,
        and populate with data
        """
        if viewer_class is None:
            return

        c = viewer_class(self._session)
        c.register_to_hub(self._session.hub)

        if data and not c.add_data(data):
            c.close(warn=False)
            return

        self.add_to_current_tab(c)
        c.show()
        return c

    @catch_error("Failed to save session")
    def save_session(self, path):
        """ Save the data collection and hub to file.

        Can be restored via restore_session

        Note: Saving of client is not currently supported. Thus,
        restoring this session will lose all current viz windows
        """
        from .glue_pickle import CloudPickler
        state = (self._data, self._hub)

        with open(path, 'w') as out:
            cp = CloudPickler(out, protocol=2)
            cp.dump(state)


    def new_tab(self):
        raise NotImplementedError()

    def add_to_current_tab(self, widget, label=None):
        raise NotImplementedError()

    def close_tab(self):
        raise NotImplementedError()

    @catch_error("Could not load data")
    def load_data(self, path):
        d = load_data(path)
        if not isinstance(d, list):
            d = [d]
        self._data.extend(d)

    def report_error(self, message, detail):
        raise NotImplementedError()

    def do(self, command):
        self._cmds.do(command)
        self._update_undo_redo_enabled()

    def undo(self):
        try:
            self._cmds.undo()
        except RuntimeError:
            pass
        self._update_undo_redo_enabled()

    def redo(self):
        try:
            self._cmds.redo()
        except RuntimeError:
            pass
        self._update_undo_redo_enabled()

    def _update_undo_redo_enabled(self):
        raise NotImplementedError()
