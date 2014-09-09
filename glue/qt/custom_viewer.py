"""
This module provides utilities for creating custom data viewers. The
goal of this module is to make it easy for users to make new
data viewers by focusing on matplotlib visualization logic,
and not UI or event processing logic.

The end user typically interacts with this code via
:func:`glue.custom_viewer`
"""
from collections import namedtuple
from inspect import getmodule

from ..clients import LayerArtist, GenericMplClient
from ..core import Data
from ..core.edit_subset_mode import EditSubsetMode
from ..core.util import nonpartial, as_list
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


def noop(*a, **k):
    return []


class AttributeInfo(namedtuple('AttributeInfo', 'id values')):

    """
    A tuple wrapping a Component of a dataset

    :param id: The identifier for this attribute (ComponentID or string)
    :param data: The numerical data (numpy array or None)
    """
    pass


class CustomViewerFactory(object):

    """
    Decorator class to build custom data viewers.

    The public methods of this class are decorators, that
    wrap custom viewer methods.

    See :func:`~glue.custom_viewer` for documentation.
    """

    def __init__(self, name, **kwargs):
        """

        :param name: The name of the custom viewer
        :type name: str

        Extra kwargs are used to specify the User interface
        """
        lbl = name.replace(' ', '')
        artist_cls = type('%sLayerArtist' % lbl, (CustomArtistBase,), {})
        client_cls = type('%sClient' % lbl, (CustomClientBase,),
                          {'artist_cls': artist_cls})

        props = CustomWidgetBase._property_set + kwargs.keys()
        widget_dict = {'client_cls': client_cls, 'LABEL': name,
                       'ui': kwargs, '_property_set': props}
        widget_dict.update(**dict((k, FormDescriptor(k))
                                  for k in kwargs))

        widget_cls = type('%sWidget' % lbl, (CustomWidgetBase,), widget_dict)
        self._artist_cls = artist_cls
        self._client_cls = client_cls
        self._widget_cls = widget_cls
        CUSTOM_WIDGETS.append(widget_cls)

        # add new classes to module namespace
        # needed for proper state saving/restoring
        for cls in [artist_cls, client_cls, widget_cls]:
            setattr(getmodule(self), cls.__name__, cls)

    def plot_subset(self, update_subset):
        """
        Decorator that wraps a function to be called to draw a subset
        """
        self._artist_cls.plot_subset = staticmethod(update_subset)
        return update_subset

    def plot_data(self, update_data):
        """
        Decorator that wraps a function to be called to draw a  dataset
        """
        self._artist_cls.plot_data = staticmethod(update_data)
        return update_data

    def setup(self, setup_func):
        """
        Decorator that wraps a function to be called when on viewer creation.
        """
        self._client_cls.setup_func = staticmethod(setup_func)
        return setup_func

    def update_settings(self, func):
        """
        Decorator that wraps a function to be called when the UI settings change.
        """
        self._widget_cls.update_settings = staticmethod(func)
        return func

    def make_selector(self, func):
        """
        Decorator that wraps a function to be called when a shape is
        drawn on the plot
        """
        self._client_cls.make_selector = staticmethod(func)
        self._widget_cls.selections_enabled = True
        return func


class CustomArtistBase(LayerArtist):

    """
    Base LayerArtist class for custom viewers
    """
    plot_data = noop
    plot_subset = noop

    def __init__(self, layer, axes, settings):
        """
        :param layer: Data or Subset object to draw
        :param axes: Matplotlib axes to use
        :param settings: dict of :class:`FormElement` instnaces
                         representing UI state
        """
        super(CustomArtistBase, self).__init__(layer, axes)
        self._settings = settings

    @property
    def settings(self):
        """
        Return a dict mapping UI keywords to current setting values
        """
        d = FormElement.dereference(self._settings, layer=self._layer)
        d['style'] = self._layer.style
        return d

    def update(self, view=None):
        """
        Redraw the layer
        """
        kwargs = self.settings
        self.clear()

        if isinstance(self._layer, Data):
            artists = self.plot_data(self._axes, **kwargs)
        else:
            artists = self.plot_subset(self._axes, **kwargs)

        for a in as_list(artists):
            a.set_zorder(self.zorder)

        self.artists = as_list(artists)


class CustomClientBase(GenericMplClient):

    """
    Base class for custom clients
    """

    # the class of LayerArtist to use
    artist_cls = None

    # custom function invoked at end of __init__
    setup_func = noop

    # custom function invoked to turn ROIs into SubsetStates
    make_selector = noop

    def __init__(self, *args, **kwargs):
        self._settings = kwargs.pop('settings', {})
        super(CustomClientBase, self).__init__(*args, **kwargs)
        self.setup_func(self.axes)

    def new_layer_artist(self, layer):
        return self.artist_cls(layer, self.axes, self._settings)

    @property
    def settings(self):
        return FormElement.dereference(self._settings)

    def apply_roi(self, roi):
        if len(self.artists) > 0:
            focus = self.artists[0].layer.data
        elif len(self.collect) > 0:
            focus = self.collect[0]
        else:
            return

        s = self.make_selector(roi, **self.settings)
        if s:
            EditSubsetMode().update(self.collect, s, focus_data=focus)

    def _update_layer(self, layer):
        for artist in self.artists[layer]:
            artist.update()

        self._redraw()


class CustomWidgetBase(DataViewer):

    """Base Qt widget class for custom viewers"""
    _property_set = DataViewer._property_set + ['redraw_on_settings_change',
                                                'selections_enabled']
    # Widget name
    LABEL = ''

    client_cls = CustomClientBase  # client class
    update_settings = noop  # custom function invoked when UI settings change
    make_selector = noop  # custom function to convert ROIs to SubsetStates

    redraw_on_settings_change = True  # redraw all layers when UI state changes?
    selections_enabled = False  # allow user to draw ROIs?

    ui = None  # dictionary that describes each UI element

    def __init__(self, session, parent=None):
        super(CustomWidgetBase, self).__init__(session, parent)
        self.central_widget = MplWidget()
        self.setCentralWidget(self.central_widget)
        self.option_widget = self.build_ui()
        self.client = self.client_cls(self._data,
                                      self.central_widget.canvas.fig,
                                      artist_container=self._container,
                                      settings=self._settings)
        self.make_toolbar()
        self.statusBar().setSizeGripEnabled(False)
        self._update_artists = []

    def options_widget(self):
        return self.option_widget

    def settings(self, layer=None):
        return FormElement.dereference(self._settings, layer)

    def build_ui(self):
        self._settings = {}
        result = QtGui.QWidget()

        layout = QtGui.QFormLayout()
        layout.setFieldGrowthPolicy(layout.AllNonFixedFieldsGrow)
        result.setLayout(layout)

        for k in sorted(type(self).ui):
            v = type(self).ui[k]
            w = FormElement.auto(v)
            w.container = self._container
            w.add_callback(self.settings_changed)
            self._settings[k] = w
            if w.ui is not None:
                layout.addRow(k.title(), w.ui)

        return result

    def settings_changed(self):
        """
        Called when UI settings change
        """
        for a in self._update_artists:
            try:
                a.remove()
            except ValueError:  # already removed
                pass
        self._update_artists = self.update_settings(self.client.axes,
                                                    **self.settings())
        if self.redraw_on_settings_change:
            self.client._update_all()

        self.client._redraw()

    def make_toolbar(self):
        result = GlueToolbar(self.central_widget.canvas, self, name=self.LABEL)
        for mode in self._mouse_modes():
            result.add_mode(mode)
        self.addToolBar(result)
        return result

    def _mouse_modes(self):
        if not self.selections_enabled:
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

        for w in self._settings.values():
            w.add_data(data)

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
        for w in self._settings.values():
            w.register_to_hub(hub)

    def unregister(self, hub):
        super(CustomWidgetBase, self).unregister(hub)
        hub.unsubscribe_all(self.client)
        hub.unsubscribe_all(self)
        for w in self._settings.values():
            hub.unsubscribe_all(w)


class FormDescriptor(object):

    def __init__(self, name):
        self.name = name

    def __get__(self, inst, owner=None):
        return inst._settings[self.name].state

    def __set__(self, inst, value):
        inst._settings[self.name].state = value


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
            return AttributeInfo(layer.data.id[cid], layer[cid])
        return AttributeInfo(cid, None)

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
            return AttributeInfo(cid, None)
        return AttributeInfo(cid, layer.data[cid])

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
