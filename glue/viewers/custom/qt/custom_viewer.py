"""
This module provides utilities for creating custom data viewers. The
goal of this module is to make it easy for users to make new
data viewers by focusing on matplotlib visualization logic,
and not UI or event processing logic.

The end user typically interacts with this code via
:func:`glue.custom_viewer`
"""

from __future__ import print_function, division


"""
Implementation notes:

Here's a high-level summary of how this code works right now:

The user creates a custom viewer using either of the following
syntaxes:


from glue import custom_viewer
my_viewer = custom_viewer('my viewer', checked=True, x='att', ...)
@my_viewer.plot_data
def plot_data(x, checked, axes):
    if checked:
        axes.plot(x)
    ...

or

from glue.viewers.custom.qt import CustomViewer
class MyViewer(CustomViewer):

    checked = True
    x = 'att'

    def plot_data(self, x, checked, axes):
        if checked:
            axes.plot(x)

This code has two "magic" features:

1. Attributes like 'checked' and 'x', passed as kwargs to custom_viewer
   or set as class-level attributes in the subclass, are turned
   into widgets based on their value

2. Functions like plot_data can take these settings as input (as well
   as some general purpose arguments like axes). Glue takes care of
   passing the proper arguments to these functions by introspecting
   their call signature. Furthermore, it extracts the current
   value of each setting (ie checked is set to True or False depending
   on what if the box is checked).

The intention of all of this magic is to let a user write "simple" functions
to draw custom plots, without having to use Glue or Qt logic directly.

Internally, Glue accomlishes this magic as follows:

 `FormElement`s are created for each attribute in (1). They build the widget
  and have a method of extracting the current value of the widget

 Functions like `plot_data` that are designed to be overriden by users
 are defined as custom descriptors -- when called at the class level,
 they become decorators that wrap and register the user-defined function.
 When called at the instance level, they become dispatch functions which
 deal with the logic in (2). The metaclass deals with registering
 UDFs when they are overridden in a subclass.
"""

from inspect import getmodule, getargspec
from types import FunctionType, MethodType
from copy import copy

import numpy as np

from glue.external import six
from qtpy import QtWidgets
from qtpy.QtCore import Qt

from glue.core.layer_artist import MatplotlibLayerArtist
from glue.config import qt_client
from glue.core import Data
from glue.core.edit_subset_mode import EditSubsetMode
from glue.utils import nonpartial, as_list, all_artists, new_artists, remove_artists
from glue import core

from glue.viewers.common.viz_client import GenericMplClient

from glue.viewers.common.qt.mpl_widget import MplWidget
from glue.viewers.common.qt.data_viewer import DataViewer
from glue.utils.qt.widget_properties import (ValueProperty, ButtonProperty,
                                             CurrentComboProperty)
from glue.viewers.common.qt.mpl_toolbar import MatplotlibViewerToolbar
from glue.viewers.common.qt.mouse_mode import PolyMode, RectangleMode

__all__ = ["AttributeInfo", "ViewerState", "UserDefinedFunction",
           "CustomViewer", "SettingsOracleInterface", "SettingsOracle",
           "CustomViewerMeta", "CustomSubsetState", "FrozenSettings",
           "CustomViewer", "CustomArtist", "CustomClient", "CustomWidgetBase",
           "FormDescriptor", "FormElement", "NumberElement", "LabeledSlider",
           "BoolElement", "FixedComponent", "ComponenentElement",
           "ChoiceElement"]


class AttributeInfo(np.ndarray):

    """
    An array subclass wrapping a Component of a dataset

    It is an array with the following additional attributes:

    * ``id``  contains the ComponentID or string name of the Component
    * ``categories`` is an array or None. For categorical Components,
      contains the distinct categories which are integer-encoded
      in the AttributeInfo
    """

    @classmethod
    def make(cls, id, values, comp, categories=None):
        values = np.asarray(values)
        result = values.view(AttributeInfo)
        result.id = id
        result.values = values
        result.categories = categories
        result._component = comp
        return result

    @classmethod
    def from_layer(cls, layer, cid, view=None):
        """
        Build an AttributeInfo out of a subset or dataset

        Parameters
        ----------
        layer : :class:`~glue.core.data.Data` or :class:`~glue.core.subset.Subset`
            The data to use
        cid : ComponentID
            The ComponentID to use
        view : numpy-style view (optional)
            What slice into the data to use
        """
        values = layer[cid, view]
        comp = layer.data.get_component(cid)
        categories = None
        if comp.categorical:
            categories = comp.categories
        return cls.make(cid, values, comp, categories)

    def __gluestate__(self, context):
        return dict(cid=context.id(self.id))

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls.make(context.object(rec['cid']), [], None)


class ViewerState(object):

    """
    Empty object for users to store data inside
    """
    pass

    def __gluestate__(self, context):
        return dict(data=[(k, context.id(v)) for k, v in self.__dict__.items()])

    @classmethod
    def __setgluestate__(cls, rec, context):
        result = cls()
        rec = rec['data']
        for k in rec:
            setattr(result, k, context.object(rec[k]))
        return result

from functools import partial


class UserDefinedFunction(object):

    """
    Descriptor to specify a UserDefinedFunction.

    Defined in CustomViewer like this::

        class CustomViewer(object):
            ...
            plot_data = UserDefinedFunction('plot_data')

    The descriptor gives CustomViewer.plot_data a dual functionality.
    When accessed at the class level, it behaves as a decorator to
    register new UDFs::

        cv = custom_viewer(...)
        @cv.plot_data  # becomes a decorator
        def plot_data_implementation(...):
            ...

    When accessed at the instance level, it becomes a dispatch function
    that calls `plot_data_implementation` with the proper arguments

    Alternatively, plot_data_implementation can be specified by
    explicitly overriding plot_data in a subclass. A metaclass
    takes care of registering the UDF in that case, so you
    can define plot_data as a normal (non-decorator, non-descriptor) method.
    """

    def __init__(self, name):
        self.name = name

    def __get__(self, instance, cls=None):
        if instance is None:
            # accessed from class level, return a decorator
            # to wrap a custom UDF
            return partial(cls._register_override_method, self.name)

        # method called at instance level,
        # return a dispatcher to the UDF
        return partial(instance._call_udf, self.name)


def introspect_and_call(func, settings):
    """
    Introspect a function for its arguments, extract values for those
    arguments from a settings oracle, and call the function

    Parameters
    ----------
    func : function
       A function to call. It should not define any keywords
    settings : SettingsOracle
       An oracle to extract values for the arguments func expects

    Returns
    -------
    The result of calling func with the proper arguments

    *Example*

    def a(x, y):
        return x, y

    introspect_and_call(a, settings) will return

    a(settings('x'), settings('y'))
    """
    a, k, _, _ = getargspec(func)

    try:
        # get the current values of each input to the UDF
        a = [settings(item) for item in a]
    except MissingSettingError as exc:
        # the UDF expects an argument that we don't know how to provide
        # try to give a helpful error message
        missing = exc.args[0]
        setting_list = "\n -".join(settings.setting_names())
        raise MissingSettingError("This custom viewer is trying to use an "
                                  "unrecognized variable named %s\n. Valid "
                                  "variable names are\n -%s" %
                                  (missing, setting_list))
    k = k or {}

    return func(*a, **k)


class SettingsOracleInterface(object):

    def __call__(self, key):
        raise NotImplementedError()

    def setting_names(self):
        return NotImplementedError()

class MissingSettingError(KeyError):
    pass

class SettingsOracle(SettingsOracleInterface):

    def __init__(self, settings, **override):

        reserved_words = set(['axes', 'layer', 'self'])
        for key in settings.keys():
            if key in reserved_words:
                raise AssertionError('You tried to create a custom setting %s' % key +
                                     ' but you cannot use a reserved word: ' +
                                     ','.join(sorted(reserved_words)))

        self.settings = settings  # dict-like, items have a value() method
        self.override = override  # look for settings here first

        # layer and view are special keywords
        self.layer = override.pop('layer', None)
        self.view = override.pop('view', None)

    def __call__(self, key):
        if key == 'self':
            return self.override['_self']
        if key in self.override:
            return self.override[key]
        if key == 'style':
            return self.layer.style
        if key == 'layer':
            return self.layer
        if key not in self.settings:
            raise MissingSettingError(key)

        return self.settings[key].value(self.layer, self.view)

    def setting_names(self):
        return list(set(list(self.settings.keys()) + ['style', 'layer']))


class CustomViewerMeta(type):

    """
    Metaclass to construct CustomViewer and subclasses

    The metaclass does two things when constructing new
    classes:

    - it finds the class-level attributes that describe
      ui elements (eg `checked=False`). It bundles these
      into a `ui` dict attribute, later used to construct
      the FormElements and widgets to represent each setting
    - It creates the qt DataViewer widget class associated with this class.
    - It looks for overridden user-defined methods like `plot_subset`,
      and registers them for later use.
    """
    def __new__(cls, name, bases, attrs):

        # don't muck with the base class
        if name == 'CustomViewer':
            return type.__new__(cls, name, bases, attrs)

        # Build UI Form
        ui = {}
        for key, value in list(attrs.items()):
            if key.startswith('_') or key in CustomViewer.__dict__:
                continue

            if not isinstance(value, (MethodType, FunctionType)):
                ui[key] = attrs.pop(key)

        attrs['ui'] = ui
        attrs.setdefault('name', name)

        # collect the UDFs
        udfs = {}

        for nm, value in list(attrs.items()):
            dscr = CustomViewer.__dict__.get(nm, None)

            if isinstance(dscr, UserDefinedFunction):
                # remove them as class method
                # register them below instead
                udfs[nm] = attrs.pop(nm)

        result = type.__new__(cls, name, bases, attrs)
        result._custom_functions = {}

        # now wrap the custom UDFs using the descriptors
        for k, v in udfs.items():
            # register UDF by mimicing the decorator syntax
            udf_decorator = getattr(result, k)
            udf_decorator(v)

        result._build_widget_subclass()

        return result


class CustomSubsetState(core.subset.SubsetState):

    """
    A SubsetState subclass that uses a CustomViewer's "select" function
    """

    def __init__(self, viewer_cls, roi, settings):
        super(CustomSubsetState, self).__init__()
        self._viewer_cls = viewer_cls
        self._settings = settings
        self._roi = roi

    def to_mask(self, data, view=None):
        settings = SettingsOracle(self._settings,
                                  layer=data, roi=self._roi, view=view)
        return introspect_and_call(self._viewer_cls._custom_functions['select'],
                                   settings)

    def copy(self):
        return CustomSubsetState(self._viewer_cls, self._roi.copy(), copy(self._settings))

    def __gluestate__(self, context):
        result = {}
        result['viewer_cls'] = self._viewer_cls.__name__
        result['settings'] = context.do(self._settings)
        result['roi'] = context.id(self._roi)
        return result

    @classmethod
    def __setgluestate__(cls, rec, context):
        viewer = getattr(getmodule(ViewerState), rec['viewer_cls'])
        settings = context.object(rec['settings'])
        roi = context.object(rec['roi'])
        return cls(viewer, roi, settings)


class FrozenSettings(object):

    """
    Encapsulates the current settings of a CustomViewer
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def value(self, key, layer=None, view=None):
        try:
            result = self.kwargs[key]
        except KeyError:
            raise MissingSettingError(key)

        if isinstance(result, AttributeInfo) and layer is not None:
            cid = result.id
            return AttributeInfo.from_layer(layer, cid, view)

        return result

    def __getitem__(self, key):

        class o(object):

            @staticmethod
            def value(layer=None, view=None):
                return self.value(key, layer, view)

        return o

    def __contains__(self, item):
        return item in self.kwargs

    def keys(self):
        return self.kwargs.keys()

    def __gluestate__(self, context):
        return dict(data=[(k, context.do(v)) for k, v in self.kwargs.items()])

    @classmethod
    def __setgluestate__(cls, rec, context):
        kwargs = dict((k, context.object(v)) for k, v in rec['data'])
        return cls(**kwargs)


@six.add_metaclass(CustomViewerMeta)
class CustomViewer(object):

    """
    Base class for custom data viewers.


    Users can either subclass this class and override
    one or more custom methods listed below, or use the
    :func:`glue.custom_viewer` function and decorate custom
    plot functions.


    *Custom Plot Methods*

    The following methods can be overridden:

     - :meth:`CustomViewer.setup`
     - :meth:`CustomViewer.plot_data`
     - :meth:`CustomViewer.plot_subset`
     - :meth:`CustomViewer.settings_changed`
     - :meth:`CustomViewer.make_selector`
     - :meth:`CustomViewer.select`

    *Method Signatures*

    Custom methods should use argument names from the following list:

      - The name of a UI element(e.g. keywords passed to :func:`glue.custom_viewer`,
        or class-level variables in subclasses). The value assigned to this
        argument will be the current UI setting (e.g. bools for checkboxes).
      - ``axes`` will contain a matplotlib Axes object
      - ``roi``  will contain the ROI a user has drawn (only available for ``make_selector``)
      - ``state`` will contain a general-purpose object to store other data
      - ``style`` contains the :class:`~glue.core.visual.VisualAttributes` describing
        a subset or dataset. Only available for ``plot_data`` and `plot_subset``
      - ``subset`` will contain the relevant :class:`~glue.core.subset.Subset` object.
         Only available for ``plot_subset``


    *Defining the UI*

    Simple widget-based UIs can be specified by providing keywords to :func:`~glue.custom_viewer`
    or class-level variables to subsets. The kind of widget to associate with each
    UI element is determined from it's type.


    *Example decorator*

    ::

        v = custom_viewer('Example', checkbox=False)

        @v.plot_data
        def plot(checkbox, axes):
            axes.plot([1, 2, 3])

    *Example subclass*

    ::

        class CustomViewerSubset(CustomViewer):
            checkbox = False

            def plot_data(self, checkbox, axes):
                axes.plot([1, 2, 3])

    The order of arguments can be listed in any order.
    """

    redraw_on_settings_change = True  #: redraw all layers when UI state changes?
    remove_artists = True             #: auto-delete artists?
    name = ''                         #: Label to give this widget in the GUI

    # hold user descriptions of desired FormElements to create
    ui = {}

    # map, e.g., 'plot_data' -> user defined function - we also make sure we
    # override this in sub-classes in CustomViewerMeta
    _custom_functions = {}

    def __init__(self, widget_instance):
        self.widget = widget_instance
        self.state = ViewerState()
        self._settings = {}

        # tracks artists created by each custom function
        self._created_artists = {}

    @property
    def selections_enabled(self):
        return 'make_selector' in self._custom_functions or 'select' in self._custom_functions

    @classmethod
    def create_new_subclass(cls, name, **kwargs):
        """
        Convenience method to build a new CustomViewer subclass

        :param name: Name of the new viewer
        :param kwargs: UI elements in the subclass
        """
        kwargs = kwargs.copy()
        kwargs['name'] = name
        # each subclass needs its own dict
        kwargs['_custom_functions'] = {}
        name = name.replace(' ', '')
        return CustomViewerMeta(name, (CustomViewer,), kwargs)

    @classmethod
    def _build_widget_subclass(cls):
        """
        Build the DataViewer subclass for this viewer
        """
        props = CustomWidgetBase._property_set + list(cls.ui.keys())
        widget_dict = {'LABEL': cls.name,
                       'ui': cls.ui,
                       'coordinator_cls': cls,
                       '_property_set': props}
        widget_dict.update(**dict((k, FormDescriptor(k))
                                  for k in cls.ui))
        widget_cls = type('%sWidget' % cls.__name__,
                          (CustomWidgetBase,),
                          widget_dict)

        cls._widget_cls = widget_cls
        qt_client.add(widget_cls)

        # add new classes to module namespace
        # needed for proper state saving/restoring
        for c in [widget_cls, cls]:
            w = getattr(getmodule(ViewerState), c.__name__, None)
            if w is not None:
                raise RuntimeError("Duplicate custom viewer detected %s" % c)

            setattr(getmodule(ViewerState), c.__name__, c)

    @classmethod
    def _register_override_method(cls, name, func):
        """
        Register a new custom method like "plot_data"

        User's need not call this directly -- it is
        called when a method is overridden or decorated
        """
        cls._custom_functions[name] = func

    def _add_data(self, data):
        for w in self._settings.values():
            w.add_data(data)

    def register_to_hub(self, hub):
        for w in self._settings.values():
            w.register_to_hub(hub)

    def unregister(self, hub):
        for w in self._settings.values():
            hub.unsubscribe_all(w)

    def _build_ui(self, callback):
        result = QtWidgets.QWidget()

        layout = QtWidgets.QFormLayout()
        layout.setFieldGrowthPolicy(layout.AllNonFixedFieldsGrow)
        result.setLayout(layout)

        for k in sorted(self.ui):
            v = self.ui[k]
            w = FormElement.auto(v)
            w.container = self.widget._layer_artist_container
            w.add_callback(callback)
            self._settings[k] = w
            if w.ui is not None:
                layout.addRow(k.title().replace('_', ' '), w.ui)

        return result

    def value(self, key, layer=None, view=None):
        return SettingsOracle(self._settings, layer=layer, view=view)(key)

    def create_axes(self, figure):
        """
        Build a new axes object
        Override for custom axes
        """
        return figure.add_subplot(1, 1, 1)

    def _build_subset_state(self, roi):

        if 'make_selector' in self._custom_functions:
            return self.make_selector(roi=roi)
        if 'select' in self._custom_functions:
            return CustomSubsetState(type(self), roi, self.settings())
        raise RuntimeError("Selection not supported for this viewer.")

    def __copy__(self):
        """
        Copying a CustomViewer freezes custom settings at their current value,
        decoupling them from future changes to the main viewer
        """
        result = type(self)(self.widget)
        result.state = copy(self.state)

        # share public attributes
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                result.__dict__[k] = v

        # copy settings
        for k in self._settings:
            result._settings[k] = copy(self._settings[k])

        return result

    def settings(self):
        """
        Return a frozen copy of the current settings of the viewer
        """
        result = {'state': copy(self.state)}
        for k in self._settings:
            result[k] = self.value(k)
        return FrozenSettings(**result)

    # List of user-defined functions.
    # Users can either use these as decorators to
    # wrap custom functions, or override them in subclasses.

    setup = UserDefinedFunction('setup')
    """
    Custom method called when plot is created
    """

    plot_subset = UserDefinedFunction('plot_subset')
    """
    Custom method called to show a subset
    """

    plot_data = UserDefinedFunction('plot_data')
    """
    Custom method called to show a dataset
    """

    make_selector = UserDefinedFunction('make_selector')
    """
    Custom method called to build a :class:`~glue.core.subset.SubsetState` from an ROI.

    See :meth:`~CustomViewer.select` for an alternative way to define selections,
    by returning Boolean arrays instead of SubsetStates.

    Functions have access to the roi by accepting an ``roi``
    argument to this function
    """

    settings_changed = UserDefinedFunction('settings_changed')
    """
    Custom method called when UI settings change.
    """

    select = UserDefinedFunction('select')
    """
    Custom method called to filter data using an ROI.

    This is an alternative function to :meth:`~CustomViewer.make_selector`,
    which returns a numpy boolean array instead of a SubsetState.

    Functions have access to the roi by accepting an ``roi``
    argument to this function
    """

    """
    End of UDF list.
    """

    def _call_udf(self, method_name, **kwargs):
        """
        Call a user-defined function stored in the _custom_functions dict

        Parameters
        ----------
        method_name : str
           The name of the user-defined method to setup a dispatch for
        **kwargs : dict
           Custom settings to pass to the UDF if they are requested by name
           as input arguments

        Returns
        -------
        The result of the UDF

        Notes
        -----
        This function builds the necessary arguments to the
        user-defined function. It also attempts to monitor
        the state of the matplotlib plot, removing stale
        artists and re-rendering the cavnas as needed.
        """

        # get the custom function
        try:
            func = self._custom_functions[method_name]
        except KeyError:
            return []

        # clear any MPL artists created on last call
        if self.remove_artists:
            layer = kwargs.get('layer', None)
            key = (layer, method_name)
            old = self._created_artists.get(key, set())
            remove_artists(old)
            current = all_artists(self.axes.figure)

        # add some extra information that the user might want
        kwargs.setdefault('_self', self)
        kwargs.setdefault('axes', self.axes)
        kwargs.setdefault('figure', self.axes.figure)
        kwargs.setdefault('state', self.state)

        # call method, keep track of newly-added artists
        settings = SettingsOracle(self._settings, **kwargs)
        result = introspect_and_call(func, settings)

        if self.remove_artists:
            new = new_artists(self.axes.figure, current)
            self._created_artists[key] = new
            if new:
                self.axes.figure.canvas.draw()
        else:
            self.axes.figure.canvas.draw()

        return result


class CustomArtist(MatplotlibLayerArtist):

    """
    LayerArtist for custom viewers
    """

    def __init__(self, layer, axes, coordinator):
        """
        :param layer: Data or Subset object to draw
        :param axes: Matplotlib axes to use
        :param settings: dict of :class:`FormElement` instnaces
                         representing UI state
        """
        super(CustomArtist, self).__init__(layer, axes)
        self._coordinator = coordinator

    def update(self, view=None):
        """
        Redraw the layer
        """
        if not self._visible:
            return

        self.clear()

        if self._coordinator.remove_artists:
            old = all_artists(self._axes.figure)

        if isinstance(self._layer, Data):
            a = self._coordinator.plot_data(layer=self._layer)
        else:
            a = self._coordinator.plot_subset(layer=self._layer, subset=self._layer)

        # if user explicitly returns the newly-created artists,
        # then use them. Otherwise, introspect to find the new artists
        if a is None:
            if self._coordinator.remove_artists:
                self.artists = list(new_artists(self._axes.figure, old))
            else:
                self.artists = []
        else:
            self.artists = as_list(a)

        for a in self.artists:
            a.set_zorder(self.zorder)


class CustomClient(GenericMplClient):

    def __init__(self, *args, **kwargs):
        self._coordinator = kwargs.pop('coordinator')
        kwargs.setdefault('axes_factory', self._coordinator.create_axes)
        super(CustomClient, self).__init__(*args, **kwargs)

        self._coordinator.axes = self.axes
        self._coordinator.setup()

    def new_layer_artist(self, layer):
        return CustomArtist(layer, self.axes, self._coordinator)

    def apply_roi(self, roi):
        if len(self.artists) > 0:
            focus = self.artists[0].layer.data
        elif len(self.collect) > 0:
            focus = self.collect[0]
        else:
            return

        s = self._coordinator._build_subset_state(roi=roi)
        if s:
            EditSubsetMode().update(self.collect, s, focus_data=focus)

    def _update_layer(self, layer):
        for artist in self.artists[layer]:
            artist.update()

        self._redraw()


class CustomWidgetBase(DataViewer):

    """Base Qt widget class for custom viewers"""

    # Widget name
    LABEL = ''

    coordinator_cls = None
    _toolbar_cls = MatplotlibViewerToolbar

    def __init__(self, session, parent=None):
        super(CustomWidgetBase, self).__init__(session, parent)
        self.central_widget = MplWidget()
        self.setCentralWidget(self.central_widget)

        self._build_coordinator()
        self.option_widget = self._build_ui()
        self.client = CustomClient(self._data,
                                   self.central_widget.canvas.fig,
                                   layer_artist_container=self._layer_artist_container,
                                   coordinator=self._coordinator)

        self.statusBar().setSizeGripEnabled(False)
        self._update_artists = []
        self.settings_changed()

    def options_widget(self):
        return self.option_widget

    def _build_coordinator(self):
        self._coordinator = self.coordinator_cls(self)

    def _build_ui(self):
        return self._coordinator._build_ui(self.settings_changed)

    def settings_changed(self):
        """
        Called when UI settings change
        """
        if self._coordinator.redraw_on_settings_change:
            self.client._update_all()

        self.client._redraw()
        self._coordinator.settings_changed()

    @property
    def tools(self):
        if self._coordinator.selections_enabled:
            return ['select:rectangle', 'select:polygon']
        else:
            return []

    def add_data(self, data):
        """Add a new data set to the widget

        :returns: True if the addition was expected, False otherwise
        """
        if data in self.client:
            return

        self.client.add_layer(data)
        self._coordinator._add_data(data)

        return True

    def add_subset(self, subset):
        """Add a subset to the widget

        :returns: True if the addition was accepted, False otherwise
        """
        self.add_data(subset.data)
        if subset.data in self.client:
            self.client.add_layer(subset)
            return True

    def register_to_hub(self, hub):
        super(CustomWidgetBase, self).register_to_hub(hub)
        self.client.register_to_hub(hub)
        self._coordinator.register_to_hub(hub)

    def unregister(self, hub):
        super(CustomWidgetBase, self).unregister(hub)
        hub.unsubscribe_all(self.client)
        hub.unsubscribe_all(self)
        self._coordinator.unregister(hub)


class FormDescriptor(object):

    def __init__(self, name):
        self.name = name

    def __get__(self, inst, owner=None):
        return inst._coordinator._settings[self.name].state

    def __set__(self, inst, value):
        inst._coordinator._settings[self.name].state = value


class FormElement(object):

    """
    Base class for user-defined settings in a custom widget.

    Each form element has a value() and a widget. Subclasses
    must override _build_ui, value, and recognizes. They
    may override register_to_hub and add_data.
    """

    def __init__(self, params):
        self.params = params
        self._callbacks = []
        self.ui = self._build_ui()
        self.container = None  # layer container

    def _build_ui(self):
        """
        Build and return a widget to represent this setting.

        The widget should automaticallhy call the changed()
        method when it's state changes
        """
        raise NotImplementedError()

    def value(self, layer=None, view=None):
        """
        Extract the value of this element

        :param layer: The Data or Subset object to use,
                      if extracting numerical data
        """
        raise NotImplementedError()

    @property
    def state(self):
        raise NotImplementedError()

    @state.setter
    def state(self, value):
        raise NotImplementedError()

    def __copy__(self):
        result = type(self)(self.params)
        result.state = self.state
        return result

    def changed(self):
        for cb in self._callbacks:
            cb()

    def add_callback(self, cb):
        """
        Register a new callback function to be invoked
        when the form state changes
        """
        self._callbacks.append(cb)

    @classmethod
    def recognizes(cls, params):
        """
        Returns whether or not a shorthand "params" object
        can be passed to __init__ to construct an element
        """
        raise NotImplementedError

    @staticmethod
    def auto(params):
        """
        Construct the appropriate FormElement subclass,
        given a shorthand object. For examle,
        FormElement.auto((0., 1.)) returns a NumberElement
        """

        def subclasses(cls):
            return cls.__subclasses__() + [g for s in cls.__subclasses__() for g in subclasses(s)]

        for cls in subclasses(FormElement):
            if cls.recognizes(params):
                return cls(params)
        raise ValueError("Unrecognzied UI Component: %s" % (params,))

    @staticmethod
    def dereference(elements, layer=None):
        """
        Given a dict of elements, extract their current settings
        into a dict

        :param elements: dict mapping labels -> FormElements
        :param layer: Subset or Data object as reference

        :reteurns: dict mapping labels -> setting value
        """
        return dict((k, v.value(layer)) for k, v in elements.items())

    def register_to_hub(self, hub):
        """
        Register the element to the hub
        """
        pass

    def add_data(self, data):
        """
        Add data to the element
        """
        pass


class NumberElement(FormElement):

    """
    A form element representing a number

    The shorthand is a tuple of 2 or 3 numbers:
    (min, max) or (min, max default)::

        e = FormElement.auto((0., 1.))
    """
    state = ValueProperty('ui')

    @classmethod
    def recognizes(cls, params):
        try:
            if len(params) not in [2, 3]:
                return False
            return all(isinstance(p, six.integer_types + (float,)) for p in params)
        except TypeError:
            return False

    def _build_ui(self):
        w = LabeledSlider(*self.params[:3])
        w.valueChanged.connect(nonpartial(self.changed))
        return w

    def value(self, layer=None, view=None):
        return self.ui.value()


class TextBoxElement(FormElement):
    """
    A form element representing a generic textbox

    The shorthand is any string starting with an _.::

        e = FormElement.auto("_default")

    Everything after the underscore is taken as the default value.
    """
    state = ValueProperty('ui')

    def _build_ui(self):
        self._widget = GenericTextBox()
        self._widget.textChanged.connect(nonpartial(self.changed))
        self.set_value(self.params[1:])
        return self._widget

    def value(self, layer=None, view=None):
        return self._widget.text()

    def set_value(self, val):
        self._widget.setText(str(val))

    @classmethod
    def recognizes(cls, params):
        try:
            if isinstance(params, str) & params.startswith('_'):
                return True
        except AttributeError:
            return None


class FloatElement(FormElement):
    """
    A form element representing a generic number box.

    The shorthand is any number::

        e = FormElement.auto(2)

    The number itself is taken as the default value.
    """
    state = ValueProperty('ui')

    def _build_ui(self):
        self._widget = GenericTextBox()
        self._widget.textChanged.connect(nonpartial(self.changed))
        self.set_value(self.params)
        return self._widget

    def value(self, layer=None, view=None):
        try:
            return float(self._widget.text())
        except ValueError:
            return None

    def set_value(self, val):
        self._widget.setText(str(val))

    @classmethod
    def recognizes(cls, params):
        return isinstance(params, (int, float)) and not isinstance(params, bool)

class GenericTextBox(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(GenericTextBox, self).__init__(parent)
        self._l = QtWidgets.QHBoxLayout()
        self._textbox = QtWidgets.QLineEdit()
        self._l.setContentsMargins(2, 2, 2, 2)
        self._l.addWidget(self._textbox)
        self.setLayout(self._l)

    @property
    def valueChanged(self):
        return self._textbox.textChanged

    @property
    def textChanged(self):
        return self._textbox.textChanged

    def value(self, layer=None, view=None):
        return self._textbox.text()

    def text(self):
        return self._textbox.text()

    def set_value(self, text):
        self._textbox.setText(text)

    setText = set_value
    setValue = set_value


class LabeledSlider(QtWidgets.QWidget):

    """
    A labeled slider widget, that handles floats and integers
    """

    def __init__(self, min, max, default=None, parent=None):
        """
        :param min: Minimum slider value
        :param max: Maximum slider value
        :param default: Initial value
        :param parent: Widget parent
        """
        super(LabeledSlider, self).__init__(parent)
        self._slider = QtWidgets.QSlider()
        self._slider.setMinimum(0)
        self._slider.setMaximum(100)
        self._slider.setOrientation(Qt.Horizontal)

        self._min = min
        self._ptp = (max - min)
        self._isint = (isinstance(min, int) and
                       isinstance(max, int) and
                       isinstance(default, (int, type(None))))

        if default is None:
            default = (min + max) / 2

        self.set_value(default)

        # setup layout
        self._lbl = QtWidgets.QLabel(str(self.value()))
        self._l = QtWidgets.QHBoxLayout()
        self._l.setContentsMargins(2, 2, 2, 2)
        self._l.addWidget(self._slider)
        self._l.addWidget(self._lbl)
        self.setLayout(self._l)

        # connect signals
        self._slider.valueChanged.connect(lambda x: self._lbl.setText(str(self.value())))

    @property
    def valueChanged(self):
        """
        Pointer to valueChanged signal.

        .. warning:: the value emitted by this signal is unscaled,
                     and shouldn't be used directly. Use .value() instead
        """
        return self._slider.valueChanged

    def value(self, layer=None, view=None):
        """
        Return the numerical value of the slider
        """
        v = self._slider.value() / 100. * self._ptp + self._min
        if self._isint:
            v = int(v)
        return v

    def set_value(self, val):
        """
        Set the numerical value of the slider
        """
        v = (1. * (val - self._min)) / self._ptp * 100
        v = min(max(int(v), 0), 100)
        self._slider.setValue(v)

    setValue = set_value


class BoolElement(FormElement):

    """
    A checkbox representing a boolean setting

    The shorthand notation is True or False::

        e = FormElement.auto(False)
    """
    state = ButtonProperty('ui')

    @classmethod
    def recognizes(cls, params):
        return isinstance(params, bool)

    def _build_ui(self):
        w = QtWidgets.QCheckBox()
        w.setChecked(self.params)
        w.toggled.connect(nonpartial(self.changed))
        return w

    def value(self, layer=None, view=None):
        return self.ui.isChecked()


class FixedComponent(FormElement):

    """
    An element for a Data Component. Does not have a widget

    The shorthand notation is 'att(comp_name)'::

        e = FormElement.auto('att(foo)')
    """

    @classmethod
    def recognizes(cls, params):
        try:
            return params.startswith('att(')
        except AttributeError:
            return False

    def _build_ui(self):
        pass

    def value(self, layer=None, view=None):
        """
        Extract the component value as an AttributeInfo object
        """
        cid = self.params.split('(')[-1][:-1]
        if layer is not None:
            cid = layer.data.id[cid]
            return AttributeInfo.from_layer(layer, cid, view)
        return AttributeInfo.make(cid, [], None)

    @property
    def state(self):
        return self.params

    @state.setter
    def state(self, value):
        self.params = value


class ComponenentElement(FormElement, core.hub.HubListener):

    """
    A dropdown selector to choose a component

    The shorthand notation is 'att'::

        e = FormElement.auto('att')
    """
    _component = CurrentComboProperty('ui')

    @property
    def state(self):
        return self._component

    @state.setter
    def state(self, value):
        self._update_components()
        if value is None:
            return
        self._component = value

    @classmethod
    def recognizes(cls, params):
        return params == 'att'

    def _build_ui(self):
        result = QtWidgets.QComboBox()
        result.currentIndexChanged.connect(nonpartial(self.changed))
        return result

    def value(self, layer=None, view=None):
        cid = self._component
        if layer is None or cid is None:
            return AttributeInfo.make(cid, [], None)
        return AttributeInfo.from_layer(layer, cid, view)

    def _list_components(self):
        """
        Determine which components to list.


        This can be overridden by subclassing to limit which components are
        visible to the user.

        """
        comps = list(set([c for l in self.container.layers
                          for c in l.data.components if not c._hidden]))
        comps = sorted(comps, key=lambda x: x.label)
        return comps

    def _update_components(self):
        combo = self.ui
        old = self._component

        combo.blockSignals(True)
        combo.clear()

        comps = self._list_components()
        for c in comps:
            combo.addItem(c.label, userData=c)

        try:
            combo.setCurrentIndex(comps.index(old))
        except ValueError:
            combo.setCurrentIndex(0)

        combo.blockSignals(False)

    def register_to_hub(self, hub):
        hub.subscribe(self, core.message.ComponentsChangedMessage,
                      nonpartial(self._update_components))

    def add_data(self, data):
        self._update_components()


class ChoiceElement(FormElement):

    """
    A dropdown selector to choose between a set of items

    Shorthand notation is a sequence of strings or a dict::

        e = FormElement.auto({'a':1, 'b':2})
        e = FormElement.auto(['a', 'b', 'c'])
    """
    state = CurrentComboProperty('ui')

    @classmethod
    def recognizes(cls, params):
        if isinstance(params, six.string_types):
            return False
        try:
            return all(isinstance(p, six.string_types) for p in params)
        except TypeError:
            return False

    def _build_ui(self):
        w = QtWidgets.QComboBox()
        for p in sorted(self.params):
            w.addItem(p)

        if isinstance(self.params, list):
            self.params = dict((p, p) for p in self.params)

        w.currentIndexChanged.connect(nonpartial(self.changed))
        return w

    def value(self, layer=None, view=None):
        return self.params[self.ui.currentText()]
