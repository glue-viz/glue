"""
This module provides utilities for creating custom data viewers. The
goal of this module is to make it easy for users to make new
data viewers by focusing on matplotlib visualization logic,
and not UI or event processing logic.

The end user typically interacts with this code via
:func:`glue.custom_viewer`
"""
from collections import namedtuple
from inspect import getmodule, getargspec
from functools import wraps
from types import FunctionType, MethodType

import numpy as np

from ..clients import LayerArtist, GenericMplClient
from ..core import Data
from ..core.edit_subset_mode import EditSubsetMode
from ..core.util import (nonpartial, as_list,
                         all_artists, new_artists, remove_artists)
from .. import core

from .widgets.data_viewer import DataViewer
from . import widget_properties as wp
from ..external.six import string_types
from ..external.qt import QtGui
from ..external.qt.QtCore import Qt
from .widgets import MplWidget
from .glue_toolbar import GlueToolbar
from .mouse_mode import PolyMode, RectangleMode


CUSTOM_WIDGETS = []


class AttributeInfo(np.ndarray):

    """
    An array subclass wrapping a Component of a dataset

    This is an array, with an extra ``id`` attribute containing the
    ComponentID or string name of the Component.
    """

    @classmethod
    def make(cls, id, values):
        result = np.asarray(values).view(AttributeInfo)
        result.id = id
        result.values = result
        return result


class ViewerState(object):

    """
    Empty object for users to store data inside
    """
    pass


def _dispatch_to_custom(method):
    """
    Method factory to build the argumenet list for a custom
    viewer function, and call it
    """

    def result(self, **kwargs):

        # get the custom function
        try:
            func = self._custom_functions[method]
        except KeyError:
            return []

        # build the argument list
        layer = kwargs.pop('layer', None)
        a, k, _, _ = getargspec(func)
        kwargs['self'] = self

        a = [kwargs.get(item) if item in kwargs
             else self._value(item, layer)
             for item in a]
        k = k or {}

        # clear any MPL artists created on last call
        if self.remove_artists:
            old = self._created_artists.get(method, set())
            remove_artists(old)
            current = all_artists(self.axes.figure)

        # call method, keep track of newly-added artists
        result = func(*a, **k)

        if self.remove_artists:
            new = new_artists(self.axes.figure, current)
            self._created_artists[method] = new

        return result

    result.__name__ = method

    return result


class CustomMeta(type):

    """
    Metaclass to construct custom viewers

    The main purpose is to detect UI form fields, and to
    build the custom widget subclass
    """
    def __new__(cls, name, bases, attrs):
        _overrideable = set(['setup', 'plot_subset', 'plot_data',
                            'settings_changed', 'make_selector'])

        if name == 'CustomViewer':
            return type.__new__(cls, name, bases, attrs)

        # Build UI Form
        ui = {}
        for key, value in attrs.items():
            if key.startswith('_') or key in CustomViewer.__dict__:
                continue

            if not isinstance(value, (MethodType, FunctionType)):
                ui[key] = attrs.pop(key)

        attrs['ui'] = ui
        attrs.setdefault('name', name)

        result = type.__new__(cls, name, bases, attrs)
        result._build_widget_subclass()

        # find and register custom viewer methods
        for name, value in attrs.items():
            if name in _overrideable:
                result._register_override_method(name, value)

        return result


class CustomViewer(object):

    """
    Base class for custom data viewers.


    Users can either subclass this class and override
    one or more custom methods listed below, or use the
    :func:`glue.custom_viewer` function to decorate custom
    plot functions.


    *Custom Plot Methods*

    The following methods can be overridden:

     - :meth:`CustomViewer.setup`
     - :meth:`CustomViewer.plot_data`
     - :meth:`CustomViewer.plot_subset`
     - :meth:`CustomViewer.settings_changed`
     - :meth:`CustomViewer.make_selector`

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

    __metaclass__ = CustomMeta

    redraw_on_settings_change = True  #: redraw all layers when UI state changes?
    remove_artists = True             #: auto-delete artists?
    name = ''                         #: Label to give this widget in the GUI

    ui = {}
    _custom_functions = {}

    def __init__(self, widget_instance):
        self.widget = widget_instance
        self.state = ViewerState()

        # tracks artists created by each custom function
        self._created_artists = {}

    @property
    def selections_enabled(self):
        return 'make_selector' in self._custom_functions

    @classmethod
    def create_new_subclass(cls, name, **kwargs):
        """
        Convenience method to build a new CustomViewer subclass

        :param name: Name of the new viewer
        :param kwargs: UI elements in the subclass
        """
        kwargs = kwargs.copy()
        kwargs['name'] = name
        name = name.replace(' ', '')
        return CustomMeta(name, (CustomViewer,), kwargs)

    @classmethod
    def _build_widget_subclass(cls):
        """
        Build the DataViewer subclass for this viewer
        """
        props = CustomWidgetBase._property_set + cls.ui.keys()
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
        CUSTOM_WIDGETS.append(widget_cls)

        # add new classes to module namespace
        # needed for proper state saving/restoring
        for cls in [widget_cls]:
            setattr(getmodule(ViewerState), cls.__name__, cls)

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
        self._settings = {}

        result = QtGui.QWidget()

        layout = QtGui.QFormLayout()
        layout.setFieldGrowthPolicy(layout.AllNonFixedFieldsGrow)
        result.setLayout(layout)

        for k in sorted(self.ui):
            v = self.ui[k]
            w = FormElement.auto(v)
            w.container = self.widget._container
            w.add_callback(callback)
            self._settings[k] = w
            if w.ui is not None:
                layout.addRow(k.title(), w.ui)

        return result

    def _value(self, setting, layer=None):
        if setting == 'style':
            if layer is None:
                raise RuntimeError("Style data not available: "
                                   "not a data or subset-specific function")
            return layer.style

        try:
            return getattr(self, setting)
        except AttributeError:
            pass

        try:
            return self._settings[setting].value(layer)
        except KeyError:
            raise AttributeError(setting)

    def create_axes(self, figure):
        """
        Build a new axes object
        Override for custom axes
        """
        return figure.add_subplot(1, 1, 1)

    @classmethod
    def setup(cls, func):
        """
        Custom method called when plot is created
        """
        cls._register_override_method('setup', func)
        return func

    @classmethod
    def plot_subset(cls, func):
        """
        Custom method called to show a subset
        """
        cls._register_override_method('plot_subset', func)
        return func

    @classmethod
    def plot_data(cls, func):
        """
        Custom method called to show a dataset
        """
        cls._register_override_method('plot_data', func)
        return func

    @classmethod
    def make_selector(cls, func):
        """
        Custom method called to build a :class:`~glue.core.subset.SubsetState` from an ROI.

        Functions have access to the roi by accepting an ``roi``
        argument to this function
        """
        cls._register_override_method('make_selector', func)
        return func

    @classmethod
    def settings_changed(cls, func):
        """
        Custom method called when UI settings change.
        """
        cls._register_override_method('settings_changed', func)
        return func

    _setup = _dispatch_to_custom('setup')
    _plot_subset = _dispatch_to_custom('plot_subset')
    _plot_data = _dispatch_to_custom('plot_data')
    _make_selector = _dispatch_to_custom('make_selector')
    _settings_changed = _dispatch_to_custom('settings_changed')


class CustomArtist(LayerArtist):

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

        old = all_artists(self._axes.figure)

        if isinstance(self._layer, Data):
            a = self._coordinator._plot_data(layer=self._layer)
        else:
            a = self._coordinator._plot_subset(layer=self._layer)

        # if user explicitly returns the newly-created artists,
        # then use them. Otherwise, introspect to find the new artists
        if a is None:
            self.artists = list(new_artists(self._axes.figure, old))
        else:
            a = as_list(a)

        for a in self.artists:
            a.set_zorder(self.zorder)


class CustomClient(GenericMplClient):

    def __init__(self, *args, **kwargs):
        self._coordinator = kwargs.pop('coordinator')
        super(CustomClient, self).__init__(*args, **kwargs)

        self._coordinator.axes = self.axes
        self._coordinator._setup()

    def new_layer_artist(self, layer):
        return CustomArtist(layer, self.axes, self._coordinator)

    def apply_roi(self, roi):
        if len(self.artists) > 0:
            focus = self.artists[0].layer.data
        elif len(self.collect) > 0:
            focus = self.collect[0]
        else:
            return

        s = self._coordinator._make_selector(roi=roi)
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

    def __init__(self, session, parent=None):
        super(CustomWidgetBase, self).__init__(session, parent)
        self.central_widget = MplWidget()
        self.setCentralWidget(self.central_widget)

        self._build_coordinator()
        self.option_widget = self._build_ui()
        self.client = CustomClient(self._data,
                                   self.central_widget.canvas.fig,
                                   artist_container=self._container,
                                   coordinator=self._coordinator)

        self.make_toolbar()
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
        self._coordinator._settings_changed()

    def make_toolbar(self):
        result = GlueToolbar(self.central_widget.canvas, self, name=self.LABEL)
        for mode in self._mouse_modes():
            result.add_mode(mode)
        self.addToolBar(result)
        return result

    def _mouse_modes(self):
        if not self._coordinator.selections_enabled:
            return []

        axes = self.client.axes

        def apply_mode(mode):
            self.client.apply_roi(mode.roi())

        # return []
        return [RectangleMode(axes, roi_callback=apply_mode),
                PolyMode(axes, roi_callback=apply_mode)]

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

    def value(self, layer=None):
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
        for cls in FormElement.__subclasses__():
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
    state = wp.ValueProperty('ui')

    @classmethod
    def recognizes(cls, params):
        try:
            if len(params) not in [2, 3]:
                return False
            return all(isinstance(p, (int, float, long)) for p in params)
        except TypeError:
            return False

    def _build_ui(self):
        w = QtGui.QSlider()
        w = LabeledSlider(*self.params[:3])
        w.valueChanged.connect(nonpartial(self.changed))
        return w

    def value(self, layer=None):
        return self.ui.value()


class LabeledSlider(QtGui.QWidget):

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
        self._slider = QtGui.QSlider()
        self._slider.setMinimum(0)
        self._slider.setMaximum(100)
        self._slider.setOrientation(Qt.Horizontal)

        self._min = min
        self._ptp = (max - min)
        if default is None:
            default = (min + max) / 2
        self._isint = (isinstance(min, int) and
                       isinstance(max, int) and
                       isinstance(default, int))

        self.set_value(default)

        # setup layout
        self._lbl = QtGui.QLabel(str(self.value()))
        self._l = QtGui.QHBoxLayout()
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
        WARNING: the value emitted by this signal is unscaled,
                 and shouldn't be used directly. Use .value() instead
        """
        return self._slider.valueChanged

    def value(self, layer=None):
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
    state = wp.ButtonProperty('ui')

    @classmethod
    def recognizes(cls, params):
        return isinstance(params, bool)

    def _build_ui(self):
        w = QtGui.QCheckBox()
        w.setChecked(self.params)
        w.toggled.connect(nonpartial(self.changed))
        return w

    def value(self, layer=None):
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

    def value(self, layer=None):
        """
        Extract the component value as an AttributeInfo object
        """
        cid = self.params.split('(')[-1][:-1]
        if layer is not None:
            return AttributeInfo.make(layer.data.id[cid], layer[cid])
        return AttributeInfo.make(cid, [])

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
    _component = wp.CurrentComboProperty('ui')

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
        result = QtGui.QComboBox()
        result.currentIndexChanged.connect(nonpartial(self.changed))
        return result

    def value(self, layer=None):
        cid = self._component
        if layer is None or cid is None:
            return AttributeInfo.make(cid, [])
        return AttributeInfo.make(cid, layer[cid])

    def _update_components(self):
        combo = self.ui
        old = self._component

        combo.blockSignals(True)
        combo.clear()

        comps = list(set([c for l in self.container.layers
                          for c in l.data.components if not c._hidden]))
        comps = sorted(comps, key=lambda x: x.label)
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
    state = wp.CurrentComboProperty('ui')

    @classmethod
    def recognizes(cls, params):
        try:
            return all(isinstance(p, string_types) for p in params)
        except TypeError:
            return False

    def _build_ui(self):
        w = QtGui.QComboBox()
        for p in self.params:
            w.addItem(p)

        if isinstance(self.params, list):
            self.params = dict((p, p) for p in self.params)

        w.currentIndexChanged.connect(nonpartial(self.changed))
        return w

    def value(self, layer=None):
        return self.params[self.ui.currentText()]
