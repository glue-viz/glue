from __future__ import absolute_import, division, print_function

import os
import sys
import imp
import logging
from collections import namedtuple
"""
Objects used to configure Glue at runtime.
"""

__all__ = ['Registry', 'SettingRegistry', 'ExporterRegistry',
           'ColormapRegistry', 'DataFactoryRegistry', 'QtClientRegistry',
           'LinkFunctionRegistry', 'LinkHelperRegistry',
           'ProfileFitterRegistry',
           'qt_client', 'data_factory', 'link_function', 'link_helper',
           'colormaps',
           'exporters', 'settings', 'fit_plugin', 'auto_refresh']


class Registry(object):

    """Container to hold groups of objects or settings.

    Registry instances are used by Glue to track objects
    used for various tasks like data linking, widget creation, etc.
    They have the following properties:

        - A `members` property, which lists each item in the registry
        - A `default_members` function, which can be overridden to lazily
          initialize the members list
        - A call interface, allowing the instance to be used as a decorator
          for users to add new items to the registry in their config files
    """

    def __init__(self):
        self._members = []
        self._loaded = False

    @property
    def members(self):
        """ A list of the members in the registry.
        The return value is a list. The contents of the list
        are specified in each subclass"""
        if not self._loaded:
            self._members = self.default_members() + self._members
            self._loaded = True

        return self._members

    def default_members(self):
        """The member items provided by default. These are put in this
        method so that code is only imported when needed"""
        return []

    def add(self, value):
        """ Add a new item to the registry """
        self._members.append(value)

    def __iter__(self):
        return iter(self.members)

    def __len__(self):
        return len(self.members)

    def __contains__(self, value):
        return value in self.members

    def __call__(self, arg):
        """This is provided so that registry instances can be used
        as decorators. The decorators should add the decorated
        code object to the registry, and return the original function"""
        self.add(arg)
        return arg


class SettingRegistry(Registry):

    """Stores key/value settings that code can use to customize Glue

    Each member is a tuple of 3 items:
      - key: the setting name [str]
      - value: the default setting [object]
      - validator: A function which tests whether the input is a valid value,
                   and raises a ValueError if invalid. On valid input,
                   returns the (possibly sanitized) setting value.
    """

    def add(self, key, value, validator=str):
        self.members.append((key, value, validator))

    def default_members(self):
        import glue.plugins  # plugins will populate this registry
        return []


class ExporterRegistry(Registry):

    """Stores functions which can export an applocation to an output file

    The members property is a list of exporters, each represented
    as a (label, save_function, can_save_function, outmode) tuple.

    save_function takes an (application, path) as input, and saves
    the session

    can_save_function takes an application as input, and raises an
    exception if saving this session is not possible

    outmode is a string, with one of 3 values:
      'file': indicates that exporter creates a file
      'directory': exporter creates a directory
      'label': exporter doesn't write to disk, but needs a label
    """

    def default_members(self):
        import glue.plugins  # discover plugins
        return []

    def add(self, label, exporter, checker, outmode='file'):
        """
        Add a new exporter
        :param label: Short label for the exporter
        :type label: str

        :param exporter: exporter function
        :type exporter: function(application, path)

        :param checker: function that checks if save is possible
        :type exporter: function(application)

        ``exporter`` should raise an exception if export isn't possible.

        :param outmode: What kind of output is created?
        :type outmode: str ('file' | 'directory' | 'label')
        """
        self.members.append((label, exporter, checker, outmode))


class ColormapRegistry(Registry):

    """Stores colormaps for the Image Viewer. The members property is
    a list of colormaps, each represented as a [name,cmap] pair.
    """

    def default_members(self):
        import matplotlib.cm as cm
        members = []
        members.append(['Gray', cm.gray])
        members.append(['Purple-Blue', cm.PuBu])
        members.append(['Yellow-Green-Blue', cm.YlGnBu])
        members.append(['Yellow-Orange-Red', cm.YlOrRd])
        members.append(['Red-Purple', cm.RdPu])
        members.append(['Blue-Green', cm.BuGn])
        members.append(['Hot', cm.hot])
        members.append(['Red-Blue', cm.RdBu])
        members.append(['Red-Yellow-Blue', cm.RdYlBu])
        members.append(['Purple-Orange', cm.PuOr])
        members.append(['Purple-Green', cm.PRGn])
        return members

    def add(self, label, cmap):
        """
        Add colormap *cmap* with label *label*.
        """
        self.members.append([label, cmap])


class DataFactoryRegistry(Registry):

    """Stores data factories. Data factories take filenames as input,
    and return :class:`~glue.core.data.Data` instances

    The members property returns a list of (function, label, identifier)
    namedtuples:

    - Function is the factory that creates the data object
    - label is a short human-readable description of the factory
    - identifier is a function that takes ``(filename, **kwargs)`` as input
      and returns True if the factory can open the file

    New data factories can be registered via::

        @data_factory('label_name', identifier, default='txt')
        def new_factory(file_name):
            ...

    This has the additional side-effect of associating
    this this factory with filenames ending in ``txt`` by default
    """
    item = namedtuple('DataFactory', 'function label identifier')

    def default_members(self):
        from .core.data_factories import __factories__
        return [self.item(f, f.label, f.identifier) for f in __factories__]

    def __call__(self, label, identifier=None, default=''):
        from .core.data_factories import set_default_factory
        if identifier is None:
            identifier = lambda *a, **k: False

        def adder(func):
            set_default_factory(default, func)
            self.add(self.item(func, label, identifier))
            return func
        return adder


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
            from .qt.widgets import default_widgets
            from .qt.custom_viewer import CUSTOM_WIDGETS
            return default_widgets + CUSTOM_WIDGETS
        except ImportError:
            logging.getLogger(__name__).warning(
                "could not import glue.qt in ConfigObject")
            return []


class LinkFunctionRegistry(Registry):

    """Stores functions to convert between quantities

    The members properety is a list of (function, info_string,
    output_labels) namedtuples. `info_string` is describes what the
    function does. `output_labels` is a list of names for each output.

    New link functions can be registered via

        @link_function(info="maps degrees to arcseconds",
                       output_labels=['arcsec'])
        def degrees2arcsec(degrees):
            return degress * 3600

    Link functions are expected to receive and return numpy arrays
    """
    item = namedtuple('LinkFunction', 'function info output_labels')

    def default_members(self):
        from .core import link_helpers
        return list(self.item(l, "", l.output_args)
                    for l in link_helpers.__LINK_FUNCTIONS__)

    def __call__(self, info="", output_labels=None):
        out = output_labels or []

        def adder(func):
            self.add(self.item(func, info, out))
            return func
        return adder


class LinkHelperRegistry(Registry):

    """Stores helper objects that compute many ComponentLinks at once

    The members property is a list of (object, info_string,
    input_labels) tuples. `Object` is the link helper. `info_string`
    describes what `object` does. `input_labels` is a list labeling
    the inputs.

    Each link helper takes a list of ComponentIDs as inputs, and
    returns an iterable object (e.g. list) of ComponentLinks.

    New helpers can be registered via

        @link_helper('Links degrees and arcseconds in both directions',
                     ['degree', 'arcsecond'])
        def new_helper(degree, arcsecond):
            return [ComponentLink([degree], arcsecond, using=lambda d: d*3600),
                    ComponentLink([arcsecond], degree, using=lambda a: a/3600)]
    """
    item = namedtuple('LinkHelper', 'helper info input_labels')

    def default_members(self):
        from .core.link_helpers import __LINK_HELPERS__ as helpers
        return list(self.item(l, l.info_text, l.input_args)
                    for l in helpers)

    def __call__(self, info, input_labels):
        def adder(func):
            self.add(self.item(func, info, input_labels))
            return func
        return adder


class ProfileFitterRegistry(Registry):
    item = namedtuple('ProfileFitter', 'cls')

    def add(self, cls):
        """
        Add colormap *cmap* with label *label*.
        """
        self.members.append(cls)

    def default_members(self):
        from .core.fitters import __FITTERS__
        return list(__FITTERS__)


class ToolRegistry(Registry):

    def default_members(self):

        from .plugins.pv_slicer import PVSlicerTool
        from .plugins.spectrum_tool import SpectrumTool
        from .qt.widgets.image_widget import ImageWidget

        return [(SpectrumTool, ImageWidget), (PVSlicerTool, ImageWidget)]

    def add(self, tool_cls, restrict_to=None):
        """
        Add a tool class to the registry, optionally specifying which widget
        class it should apply to (``restrict_to``). if ``restrict_to`` is set
        to `None`, the tool applies to all classes.
        """
        self.members.append((tool_cls, restrict_to))

    def get_tools(self, requested_widget_cls, allow_subclass=True):

        tools = []
        for (tool_cls, widget_cls) in self.members:

            if widget_cls is None or widget_cls is requested_widget_cls:
                tools.append(tool_cls)
            elif allow_subclass:
                if issubclass(requested_widget_cls, widget_cls):
                    tools.append(tool_cls)

        return tools


class BooleanSetting(object):

    def __init__(self, default=True):
        self.state = default

    def __call__(self, state=None):
        if state not in [None, True, False]:
            raise ValueError("Invalid True/False setting: %s" % state)

        if state is not None:
            self.state = state

        return self.state

qt_client = QtClientRegistry()
data_factory = DataFactoryRegistry()
link_function = LinkFunctionRegistry()
link_helper = LinkHelperRegistry()
colormaps = ColormapRegistry()
exporters = ExporterRegistry()
settings = SettingRegistry()
fit_plugin = ProfileFitterRegistry()
tool_registry = ToolRegistry()

# watch loaded data files for changes?
auto_refresh = BooleanSetting(False)


def load_configuration(search_path=None):
    ''' Find and import a config.py file

    Returns:

       The module object

    Raises:

       Exception, if no module was found
    '''
    search_order = search_path or _default_search_order()
    result = imp.new_module('config')

    for config_file in search_order:
        dir = os.path.dirname(config_file)
        try:
            sys.path.append(dir)
            config = imp.load_source('config', config_file)
            result = config
        except IOError:
            pass
        except Exception as e:
            raise type(e), "Error loading config file %s:\n%s" % (config_file, e), sys.exc_info()[2]
        finally:
            sys.path.remove(dir)

    return result


def _default_search_order():
    """
    The default configuration file search order:

       * current working directory
       * environ var GLUERC
       * HOME/.glue/config.py
       * Glue's own default config
    """
    search_order = [os.path.join(os.getcwd(), 'config.py')]
    if 'GLUERC' in os.environ:
        search_order.append(os.environ['GLUERC'])
    search_order.append(os.path.expanduser('~/.glue/config.py'))
    return search_order[::-1]
