import logging

from ..core.config import Registry

__all__ = ['QtClientRegistry', 'QtToolRegistry', 'qt_client', 'tool_registry']


class QtClientRegistry(Registry):

    """Stores QT widgets to visualize data.

    The members property is a list of Qt widget classes

    New widgets can be registered via::

        @qt_client
        class CustomWidget(QMainWindow):
            ...
    """

    def default_members(self):
        try:
            from .widgets import default_widgets
            from .custom_viewer import CUSTOM_WIDGETS
            return default_widgets + CUSTOM_WIDGETS
        except ImportError:
            logging.getLogger(__name__).warning(
                "could not import glue.qt in ConfigObject")
            return []


class QtToolRegistry(Registry):

    def __init__(self):
        self._members = {}
        self._lazy_members = []
        self._loaded = False

    @property
    def members(self):
        self._load_lazy_members()
        if not self._loaded:
            defaults = self.default_members()
            for key in defaults:
                if key in self._members:
                    self._members[key].extend(defaults[key])
                else:
                    self._members[key] = defaults[key]
            self._loaded = True
        return self._members

    def default_members(self):
        defaults = {}
        for viewer in qt_client.members:
            try:
                defaults[viewer] = viewer._get_default_tools()
            except AttributeError:
                logging.getLogger(__name__).warning(
                    "could not get default tools for {0}".format(viewer.__name__))
                defaults[viewer] = []

        return defaults

    def add(self, tool_cls, widget_cls=None):
        """
        Add a tool class to the registry, optionally specifying which widget
        class it should apply to (``widget_cls``). if ``widget_cls`` is set
        to `None`, the tool applies to all classes.
        """
        if widget_cls in self.members:
            self.members[widget_cls].append(tool_cls)
        else:
            self.members[widget_cls] = [tool_cls]
            
qt_client = QtClientRegistry()
tool_registry = QtToolRegistry()
