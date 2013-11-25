from functools import wraps
import traceback
from collections import namedtuple

from .hub import HubListener, Hub
from .data_collection import DataCollection
from .data_factories import load_data
from .command import CommandStack

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
        context = namedtuple('GlueContext', 'data_collection application')
        context = context(data_collection=self._data, application=self)
        self._cmds = CommandStack(context)


    def new_data_viewer(self, viewer_class, data=None):
        """
        Create a new data viewer, add it to the UI,
        and populate with data
        """
        raise NotImplementedError()

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

    def undo(self):
        self._cmds.undo()

    def redo(self):
        self._cmds.redo()
