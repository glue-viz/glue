import os
import sys
import imp
import logging
from collections import namedtuple


class Registry(object):
    """Registry instances are used by Glue to track objects
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
            self._members.extend(self.default_members())
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

    def __contains__(self, value):
        return value in self.members

    def __call__(self, arg):
        """This is provided so that registry instances can be used
        as decorators. The decorators should add the decorated
        code object to the registry, and return the original function"""
        self.add(arg)
        return arg


class DataFactoryRegistry(Registry):
    """Stores data factories. Data factories take filenames as input,
    and return :class:`~glue.core.Data` instances

    The members property returns a list of (function, label, filter)
    namedtuples. Filter specifies what kind of file extensions
    the factory can open

    New data factories can be registered via:

        @data_factory('label_name', '*.txt')
        def new_factory(file_name):
            ...
    """
    item = namedtuple('DataFactory', 'function label filter')

    def default_members(self):
        from .core.data_factories import __factories__
        return [self.item(f, f.label, f.file_filter) for f in __factories__]

    def __call__(self, label, fltr):
        def adder(func):
            self.add(self.item(func, label, fltr))
            return func
        return adder


class QtClientRegistry(Registry):
    """Stores QT widgets to visualize data.

    The members property is a list of Qt widget classes

    New widgets can be registered via:

        @qt_client
        class CustomWidget(QMainWindow):
            ...
    """
    def default_members(self):
        try:
            from .qt.widgets.scatter_widget import ScatterWidget
            from .qt.widgets.image_widget import ImageWidget
            from .qt.widgets.histogram_widget import HistogramWidget
            from .qt.widgets.wwt_widget import WWTWidget
            return [ScatterWidget, ImageWidget, HistogramWidget, WWTWidget]
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


qt_client = QtClientRegistry()
data_factory = DataFactoryRegistry()
link_function = LinkFunctionRegistry()
link_helper = LinkHelperRegistry()


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
        try:
            config = imp.load_source('config', config_file)
            result = config
        except IOError:
            pass
        except Exception as e:
            raise Exception("Error loading config file %s:\n%s" %
                            (config_file, e))

    return result


def _default_search_order():
    """
    The default configuration file search order:

       * current working directory
       * environ var GLUERC
       * HOME/.glue/config.py
       * Glue's own default config
    """
    current_module = sys.modules['glue'].__path__[0]
    search_order = [os.path.join(os.getcwd(), 'config.py')]
    if 'GLUERC' in os.environ:
        search_order.append(os.environ['GLUERC'])
    search_order.append(os.path.expanduser('~/.glue/config.py'))
    return search_order[::-1]
