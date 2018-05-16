from __future__ import absolute_import, division, print_function

import os
import imp
import sys
from collections import namedtuple

"""
Objects used to configure Glue at runtime.
"""

__all__ = ['Registry', 'SettingRegistry', 'ExporterRegistry',
           'ColormapRegistry', 'DataFactoryRegistry', 'QtClientRegistry',
           'LinkFunctionRegistry', 'LinkHelperRegistry', 'ViewerToolRegistry',
           'LayerActionRegistry', 'ProfileFitterRegistry', 'qt_client', 'data_factory',
           'link_function', 'link_helper', 'colormaps', 'exporters', 'settings',
           'fit_plugin', 'auto_refresh', 'importer', 'DictRegistry',
           'preference_panes', 'PreferencePanesRegistry',
           'DataExporterRegistry', 'data_exporter', 'layer_action',
           'SubsetMaskExporterRegistry', 'SubsetMaskImporterRegistry',
           'StartupActionRegistry', 'startup_action', 'QtFixedLayoutTabRegistry',
           'qt_fixed_layout_tab', 'KeyboardShortcut', 'keyboard_shortcut']


CFG_DIR = os.path.join(os.path.expanduser('~'), '.glue')


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
        self._lazy_members = []
        self._loaded = False

    @property
    def members(self):
        """ A list of the members in the registry.
        The return value is a list. The contents of the list
        are specified in each subclass"""
        self._load_lazy_members()
        if not self._loaded:
            self._members = self.default_members() + self._members
            self._loaded = True

        return self._members

    def default_members(self):
        """The member items provided by default. These are put in this
        method so that code is only imported when needed"""
        return []

    def add(self, value):
        """
        Add a new item to the registry.
        """
        self._members.append(value)

    def lazy_add(self, value):
        """
        Add a reference to a plugin which will be loaded when needed.
        """
        self._lazy_members.append(value)

    def _load_lazy_members(self):
        from glue.plugins import load_plugin
        while self._lazy_members:
            plugin = self._lazy_members.pop()
            load_plugin(plugin)

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


class DictRegistry(Registry):
    """
    Base class for registries that are based on dictionaries instead of lists
    of objects.
    """

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
        return {}


class SettingRegistry(DictRegistry):

    """Stores key/value settings that code can use to customize Glue

    Each member is a tuple of 3 items:
      - key: the setting name [str]
      - value: the default setting [object]
      - validator: A function which tests whether the input is a valid value,
                   and raises a ValueError if invalid. On valid input,
                   returns the (possibly sanitized) setting value.
    """

    def __init__(self):
        super(SettingRegistry, self).__init__()
        self._validators = {}
        self._defaults = {}

    def add(self, key, default=None, validator=None):

        if validator is None:
            validator = lambda x: x

        self._defaults[key] = validator(default)
        self._validators[key] = validator

    def __getattr__(self, attr):
        if attr.startswith('_'):
            raise AttributeError("No such setting: {0}".format(attr))
        else:
            if attr in self._members:
                return self._members[attr]
            elif attr in self._defaults:
                return self._defaults[attr]
            else:
                raise AttributeError("No such setting: {0}".format(attr))

    def __setattr__(self, attr, value):
        if attr.startswith('_'):
            object.__setattr__(self, attr, value)
        elif attr in self:
            self._members[attr] = self._validators[attr](value)
        else:
            raise AttributeError("No such setting: {0}".format(attr))

    def __dir__(self):
        return sorted(self._members.keys())

    def __contains__(self, setting):
        return setting in self._defaults

    def __iter__(self):
        for key in self._defaults:
            value = self._members.get(key, self._defaults[key])
            yield key, value, self._validators[key]

    def __str__(self):
        s = ""
        for name, value, validator in self:
            s += "{0}: {1}\n".format(name, value)
        return s

    def reset_defaults(self):
        self._members.clear()

    def is_default(self, setting):
        return setting in self._defaults and setting not in self._members


class QGlueParserRegistry(Registry):
    """
    Registry for parsers that can be used to interpret arguments to the
    :func:`~glue.qglue` function.

    The members property is a list of parsers, each represented as a named tuple
    with ``data_class``, ``parser`` and ``priority`` attributes, where ``class``
    defines the class for which to use the parser, and ``parser`` is a function
    that takes the input data and returns a list of glue
    :class:`~glue.core.Data` objects. The ``parser`` functions should take two
    arguments: the variable containing the data being parsed, and a label. In
    addition, the priority (defaulting to 0) can be specified in case one wants
    to make sure sub-classes get tested before more general classes. The
    priority should be a numerical value, and the larger it is the higher the
    priority.
    """

    item = namedtuple('DataFactory', 'data_class parser priority')

    def add(self, data_class, parser, priority=0):
        """
        Add a new parser

        Parameters
        ----------
        data_class : class
            The type of of data for which to use the specified parser
        parser : func
            The function to use to parse the input data
        priority : int, optional
            The priority, which is used to determine the order in which to check
            the parsers.
        """
        self.members.append(self.item(data_class, parser, priority))

    def __call__(self, data_class, priority=0):
        def adder(func):
            if isinstance(data_class, tuple):
                for dc in data_class:
                    self.add(dc, func, priority=priority)
            else:
                self.add(data_class, func, priority=priority)
            return func
        return adder

    def __iter__(self):
        for member in sorted(self.members, key=lambda x: -x.priority):
            yield member


class DataImportRegistry(Registry):
    """
    Stores functions which can import data.

    The members property is a list of importers, each represented as a
    ``(label, load_function)`` tuple. The ``load_function`` should take no
    arguments and return a list of :class:`~glue.core.data.Data` objects.
    """

    def add(self, label, importer):
        """
        Add a new importer
        :param label: Short label for the importer
        :type label: str

        :param importer: importer function
        :type importer: function()
        """
        self.members.append((label, importer))

    def __call__(self, label):
        def adder(func):
            self.add(label, func)
            return func
        return adder


class MenubarPluginRegistry(Registry):
    """
    Stores menubar plugins.

    The members property is a list of menubar plugins, each represented as a
    ``(label, function)`` tuple. The ``function`` should take two items which
    are a reference to the session and to the data collection respectively.
    """

    def add(self, label, function):
        """
        Add a new menubar plugin
        :param label: Short label for the plugin
        :type label: str

        :param function: function
        :type function: function()
        """
        self.members.append((label, function))

    def __call__(self, label):
        def adder(func):
            self.add(label, func)
            return func
        return adder


class PreferencePanesRegistry(DictRegistry):
    """
    Stores preference panes

    The members property is a list of tuples of Qt widget classes that can have
    their own tab in the preferences window.
    """

    def add(self, label, widget_cls):
        self._members[label] = widget_cls

    def __iter__(self):
        for label in self._members:
            yield label, self._members[label]


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

    def add(self, label, exporter, checker, outmode=None):
        """
        Add a new exporter

        Parameters
        ----------
        label : str
            Short label for the exporter

        exporter : func
            Exporter function which takes two arguments: the application and
            optionally the path or label to create. This function should raise
            an exception if export isn't possible.

        checker : func
            Function that checks if saving is possible, which takes one
            argument: the application.

        outmode : str or `None`
            Indicates what kind of output is created. This can be either set to
            ``'file'``, ``'directory'``, ``'label'``, or `None`.
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
        members.append(['Viridis', cm.viridis])
        members.append(['Plasma', cm.plasma])
        members.append(['Inferno', cm.inferno])
        members.append(['Magma', cm.magma])
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

    def __getitem__(self, cmap_name):
        for name, cmap in self.members:
            if name == cmap_name:
                return cmap
        raise KeyError(cmap_name)

    def name_from_cmap(self, cmap_desired):
        for name, cmap in self.members:
            if cmap is cmap_desired:
                return name
        raise ValueError("Could not find name for colormap")


class DataFactoryRegistry(Registry):

    """Stores data factories. Data factories take filenames as input,
    and return :class:`~glue.core.data.Data` instances

    The members property returns a list of (function, label, identifier,
    priority) namedtuples:

    - Function is the factory that creates the data object
    - label is a short human-readable description of the factory
    - identifier is a function that takes ``(filename, **kwargs)`` as input
      and returns True if the factory can open the file
    - priority is a numerical value that indicates how confident the data
      factory is that it should read the data, relative to other data
      factories. For example, a highly specialized FITS reader for specific
      FITS file types can be given a higher priority than the generic FITS
      reader in order to take precedence over it.

    New data factories can be registered via::

        @data_factory('label_name', identifier=identifier, priority=10)
        def new_factory(file_name):
            ...

    If not specified, the priority defaults to 0.
    """

    item = namedtuple('DataFactory', 'function label identifier priority deprecated')

    def __call__(self, label, identifier=None, priority=None, default='', deprecated=False):

        if identifier is None:
            identifier = lambda *a, **k: False

        if priority is None:
            if deprecated:
                priority = -1000
            else:
                priority = 0

        def adder(func):
            self.add(self.item(func, label, identifier, priority, deprecated))
            return func

        return adder

    def __iter__(self):
        for member in sorted(self.members, key=lambda x: (-x.priority, x.label)):
            yield member


class DataExporterRegistry(Registry):
    """
    Stores data exporters. Data exporters take a data/subset object as input
    followed by a filename.
    """

    item = namedtuple('DataFactory', 'function label extension')

    def __call__(self, label, extension=[]):
        def adder(func):
            self.add(self.item(func, label, extension))
            return func
        return adder

    def __iter__(self):
        for member in sorted(self.members, key=lambda x: x.label):
            yield member


class SubsetMaskExporterRegistry(DataExporterRegistry):
    """
    Stores mask exporters. Mask exporters should take a filename followed by
    a dictionary of Numpy boolean arrays all with the same dimensions.
    """
    item = namedtuple('SubsetMaskExporter', 'function label extension')


class SubsetMaskImporterRegistry(DataExporterRegistry):
    """
    Stores mask importers. Mask importers should take a filename and return a
    dictionary of Numpy boolean arrays.
    """
    item = namedtuple('SubsetMaskImporter', 'function label extension')


class QtClientRegistry(Registry):
    """
    Stores QT widgets to visualize data.

    The members property is a list of Qt widget classes

    New widgets can be registered via::

        @qt_client
        class CustomWidget(QMainWindow):
            ...
    """


class QtFixedLayoutTabRegistry(Registry):
    """
    Stores Qt pre-defined tabs (non-MDI)

    New widgets can be registered via::

        @qt_fixed_layout_tab
        class CustomTab(QWidget):
            ...
    """


class ViewerToolRegistry(DictRegistry):

    def add(self, tool_cls):
        """
        Add a tool class to the registry. The the ``tool_id`` attribute on the
        tool_cls should be set, and is used by the viewers to indicate which
        tools they want to
        """
        if tool_cls.tool_id in self.members:
            raise ValueError("Tool ID '{0}' already registered".format(tool_cls.tool_id))
        else:
            self.members[tool_cls.tool_id] = tool_cls

    def __call__(self, tool_cls):
        self.add(tool_cls)
        return tool_cls


class StartupActionRegistry(DictRegistry):

    def add(self, startup_name, startup_function):
        """
        Add a startup function to the registry. This is a function that will
        get called once glue has been started and any data loaded, and can
        be used to set up specific layouts and create links.

        Startup actions are triggered by either specifying comma-separated names
        of actions on the command-line::

            glue --startup=mystartupaction

        or by passing an iterable of startup action names to the ``startup``
        keyword of ``GlueApplication``.

        The startup function will be given the session object and the data
        collection object.
        """
        if startup_name in self.members:
            raise ValueError("A startup action with the name '{0}' already exists".format(startup_name))
        else:
            self.members[startup_name] = startup_function

    def __call__(self, name):
        def adder(func):
            self.add(name, func)
            return func
        return adder


class LinkFunctionRegistry(Registry):

    """Stores functions to convert between quantities

    The members properety is a list of (function, info_string,
    output_labels) namedtuples. ``info_string`` describes what the
    function does. ``output_labels`` is a list of names for each output.
    ``category`` is a category in which the link funtion will appear (defaults
    to 'General').

    New link functions can be registered via

        @link_function(info="maps degrees to arcseconds",
                       output_labels=['arcsec'])
        def degrees2arcsec(degrees):
            return degress * 3600

    Link functions are expected to receive and return numpy arrays
    """
    item = namedtuple('LinkFunction', 'function info output_labels category')

    def __call__(self, info="", output_labels=None, category='General'):
        out = output_labels or []

        def adder(func):
            self.add(self.item(func, info, out, category))
            return func
        return adder


class LayerActionRegistry(Registry):
    """
    Stores custom menu actions available when the user select one or more
    datasets, subset group, or subset in the data collection view.

    This members property is a list of named tuples with the following
    attributes:

    * ``label``: the user-facing name of the action
    * ``tooltip``: the text that appears when hovering with the mouse over the action
    * ``callback``: the function to call when the action is triggered
    * ``icon``: an icon image to use for the layer action
    * ``single``: whether to show this action only when selecting single layers (default: `False`)
    * ``data``: if ``single`` is `True` whether to only show the action when selecting a dataset
    * ``subset_group``: if ``single`` is `True` whether to only show the action when selecting a subset group
    * ``subset``: if ``single`` is `True` whether to only show the action when selecting a subset

    The callback function is called with two arguments. If ``single`` is
    `True`, the first argument is the selected layer, otherwise it is the list
    of selected layers. The second argument is the
    `~glue.core.data_collection.DataCollection` object.
    """
    item = namedtuple('LayerAction', 'label tooltip callback icon single data subset_group, subset')

    def __call__(self, label, callback=None, tooltip=None, icon=None, single=False,
                 data=False, subset_group=False, subset=False):

        # Backward-compatibility
        if callback is not None:
            self.add(self.item(label, tooltip, callback, icon, True,
                               False, False, True))
            return True

        def adder(func):
            self.add(self.item(label, tooltip, func, icon, single,
                               data, subset_group, subset))
            return func
        return adder


class LinkHelperRegistry(Registry):

    """Stores helper objects that compute many ComponentLinks at once

    The members property is a list of (object, info_string,
    input_labels) tuples. `Object` is the link helper. `info_string`
    describes what `object` does. `input_labels` is a list labeling
    the inputs. ``category`` is a category in which the link funtion will appear
    (defaults to 'General').

    Each link helper takes a list of ComponentIDs as inputs, and
    returns an iterable object (e.g. list) of ComponentLinks.

    New helpers can be registered via

        @link_helper('Links degrees and arcseconds in both directions',
                     ['degree', 'arcsecond'])
        def new_helper(degree, arcsecond):
            return [ComponentLink([degree], arcsecond, using=lambda d: d*3600),
                    ComponentLink([arcsecond], degree, using=lambda a: a/3600)]
    """
    item = namedtuple('LinkHelper', 'helper info input_labels category')

    def __call__(self, info, input_labels, category='General'):
        def adder(func):
            self.add(self.item(func, info, input_labels, category))
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
        from glue.core.fitters import __FITTERS__
        return list(__FITTERS__)


class BooleanSetting(object):

    def __init__(self, default=True):
        self.state = default

    def __call__(self, state=None):
        if state not in [None, True, False]:
            raise ValueError("Invalid True/False setting: %s" % state)

        if state is not None:
            self.state = state

        return self.state


class KeyboardShortcut(DictRegistry):
    """
    Stores keyboard shortcuts.
    The members property is a dictionary within a dictionary of keyboard
    shortcuts, which is represented as (viewer,(keybind,function)). The
    ``function`` should take one item, which is a reference to the session.
    """

    def add(self, valid_viewers, keybind, function):
        """
        Add a new keyboard shortcut

        Parameters
        ----------
        arg1: list
            list of viewers where event can be fired
        arg2: Qt.Key
            type of key event
        arg3: function()
            function to be run that corresponds with key
        """
        if valid_viewers:
            for viewer in valid_viewers:
                if viewer in self.members:
                    if keybind in self.members[viewer]:
                        raise ValueError("Keyboard shortcut '{0}' already registered in {1}".format(keybind, viewer))
                    else:
                        self.members[viewer][keybind] = function
                else:
                    self.members[viewer] = {keybind: function}
        else:
            if None in self.members:
                if keybind in self.members[None]:
                    raise ValueError("Keyboard shortcut '{0}' already registered in {1}".format(keybind, None))
                else:
                    self.members[None][keybind] = function
            else:
                self.members[None] = {keybind: function}

    def __call__(self, keybind, valid_viewers):
        def adder(func):
            self.add(valid_viewers, keybind, func)
            return func
        return adder


qt_client = QtClientRegistry()
qt_fixed_layout_tab = QtFixedLayoutTabRegistry()
viewer_tool = ViewerToolRegistry()
link_function = LinkFunctionRegistry()
link_helper = LinkHelperRegistry()
colormaps = ColormapRegistry()
importer = DataImportRegistry()
exporters = ExporterRegistry()
settings = SettingRegistry()
fit_plugin = ProfileFitterRegistry()
layer_action = LayerActionRegistry()
menubar_plugin = MenubarPluginRegistry()
preference_panes = PreferencePanesRegistry()
qglue_parser = QGlueParserRegistry()
startup_action = StartupActionRegistry()
keyboard_shortcut = KeyboardShortcut()

# watch loaded data files for changes?
auto_refresh = BooleanSetting(False)
enable_contracts = BooleanSetting(False)

# Data and subset I/O
data_factory = DataFactoryRegistry()
data_exporter = DataExporterRegistry()
subset_mask_exporter = SubsetMaskExporterRegistry()
subset_mask_importer = SubsetMaskImporterRegistry()

# Backward-compatibility
single_subset_action = layer_action


def load_configuration(search_path=None):
    """
    Find and import a config.py file

    Returns:

       The module object

    Raises:

       Exception, if no module was found
    """
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
            raise type(e)("Error loading config file %s:\n%s" % (config_file, e), sys.exc_info()[2])
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
    search_order.append(os.path.join(CFG_DIR, 'config.py'))
    return search_order[::-1]


# ##### Now define global settings ######

GRAY = '#7F7F7F'
BLUE = "#1F78B4"
GREEN = "#33A02C"
RED = "#E31A1C"
ORANGE = "#FF7F00"
PURPLE = "#6A3D9A"
YELLOW = "#FFFF99"
BROWN = "#8C510A"
PINK = "#FB9A99"
LIGHT_BLUE = "#A6CEE3"
LIGHT_GREEN = "#B2DF8A"
LIGHT_RED = "#FB9A99"
LIGHT_ORANGE = "#FDBF6F"
LIGHT_PURPLE = "#CAB2D6"

settings.add('SUBSET_COLORS', [RED, GREEN, BLUE, BROWN, ORANGE, PURPLE, PINK], validator=list)
settings.add('DATA_COLOR', '#595959')
settings.add('DATA_ALPHA', 0.8, validator=float)
settings.add('BACKGROUND_COLOR', '#FFFFFF')
settings.add('FOREGROUND_COLOR', '#000000')
settings.add('SHOW_LARGE_DATA_WARNING', True, validator=bool)
settings.add('SHOW_INFO_PROFILE_OPEN', True, validator=bool)
settings.add('SHOW_WARN_PROFILE_DUPLICATE', True, validator=bool)
