"""
This module provides utilities for creating custom data viewers. The goal of
this module is to make it easy for users to make new data viewers by focusing on
matplotlib visualization logic, and not UI or event processing logic.

The end user typically interacts with this code via :func:`glue.custom_viewer`
"""

# Implementation notes:
#
# Here's a high-level summary of how this code works right now:
#
# The user creates a custom viewer using either of the following
# syntaxes:
#
#
# from glue import custom_viewer
# my_viewer = custom_viewer('my viewer', checked=True, x='att', ...)
# @my_viewer.plot_data
# def plot_data(x, checked, axes):
#     if checked:
#         axes.plot(x)
#     ...
#
# or
#
# from glue.viewers.custom.qt import CustomViewer
# class MyViewer(CustomViewer):
#
#     checked = True
#     x = 'att'
#
#     def plot_data(self, x, checked, axes):
#         if checked:
#             axes.plot(x)
#
# This code has two "magic" features:
#
# 1. Attributes like 'checked' and 'x', passed as kwargs to custom_viewer
#    or set as class-level attributes in the subclass, are turned
#    into widgets based on their value
#
# 2. Functions like plot_data can take these settings as input (as well
#    as some general purpose arguments like axes). Glue takes care of
#    passing the proper arguments to these functions by introspecting
#    their call signature. Furthermore, it extracts the current
#    value of each setting (ie checked is set to True or False depending
#    on what if the box is checked).
#
# The intention of all of this magic is to let a user write "simple" functions
# to draw custom plots, without having to use Glue or Qt logic directly.
#
# Internally, Glue accomlishes this magic as follows:
#
#  `FormElement`s are created for each attribute in (1). They build the widget
#   and have a method of extracting the current value of the widget
#
#  Functions like `plot_data` that are designed to be overriden by users
#  are defined as custom descriptors -- when called at the class level,
#  they become decorators that wrap and register the user-defined function.
#  When called at the instance level, they become dispatch functions which
#  deal with the logic in (2). The metaclass deals with registering
#  UDFs when they are overridden in a subclass.

from inspect import getmodule
from functools import partial

from inspect import getfullargspec

from types import FunctionType, MethodType

import numpy as np

from qtpy.QtWidgets import QWidget, QGridLayout, QLabel

from echo.qt import autoconnect_callbacks_to_qt

from glue.config import qt_client

from glue.core import BaseData
from glue.core.subset import SubsetState
from glue.core.data_combo_helper import ComponentIDComboHelper
from glue.core.component_id import ComponentID

from glue.utils import as_list, all_artists, new_artists, categorical_ndarray, defer_draw

from glue.viewers.matplotlib.qt.data_viewer import MatplotlibDataViewer
from glue.viewers.matplotlib.state import MatplotlibDataViewerState, MatplotlibLayerState
from glue.viewers.matplotlib.layer_artist import MatplotlibLayerArtist

from glue.viewers.custom.qt.elements import (FormElement,
                                             DynamicComponentIDProperty,
                                             FixedComponentIDProperty)

__all__ = ["AttributeWithInfo", "ViewerUserState", "UserDefinedFunction",
           "CustomViewer", "CustomViewerMeta", "CustomSubsetState",
           "CustomViewer", "CustomLayerArtist", "CustomMatplotlibDataViewer"]


class AttributeWithInfo(np.ndarray):
    """
    An array subclass wrapping a Component of a dataset It is an array with the
    following additional attributes: ``id``  contains the ComponentID or string
    name of the Component, and ``categories`` is an array or `None`. For
    categorical Components, it contains the distinct categories which are
    integer-encoded in the AttributeWithInfo
    """

    @classmethod
    def make(cls, id, values, categories=None):
        values = np.asarray(values)
        result = values.view(AttributeWithInfo)
        result.id = id
        result.values = values
        result.categories = categories
        return result

    @classmethod
    def from_layer(cls, layer, cid, view=None):
        """
        Build an AttributeWithInfo out of a subset or dataset.

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
        if isinstance(values, categorical_ndarray):
            categories = values.categories
            values = values.codes
        else:
            categories = None
        return cls.make(cid, values, categories)

    def __gluestate__(self, context):
        return dict(cid=context.id(self.id))

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls.make(context.object(rec['cid']), [], None)


class ViewerUserState(object):
    """
    Empty object for users to store data inside.
    """

    def __gluestate__(self, context):
        return dict(data=[(k, context.id(v)) for k, v in self.__dict__.items()])

    @classmethod
    def __setgluestate__(cls, rec, context):
        result = cls()
        rec = rec['data']
        for k in rec:
            setattr(result, k, context.object(rec[k]))
        return result


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

    Alternatively, plot_data_implementation can be specified by explicitly
    overriding plot_data in a subclass. A metaclass takes care of registering
    the UDF in that case, so you can define plot_data as a normal
    (non-decorator, non-descriptor) method.
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


def introspect_and_call(func, state, override):
    """
    Introspect a function for its arguments, extract values for those
    arguments from a state class, and call the function

    Parameters
    ----------
    func : function
        A function to call. It should not define any keywords
    state : State
        A state class containing the values to pass
    override : dict
        A dictionary containing values that should override the state

    Returns
    -------
    The result of calling func with the proper arguments

    *Example*

    def a(x, y):
        return x, y

    introspect_and_call(a, state) will return

    a(state.x, state.y)

    Attributes will be used from ``override`` before ``state``.
    """

    a, k = getfullargspec(func)[:2]

    args = []
    for item in a:
        if item in override:
            args.append(override[item])
        elif hasattr(state, item):
            args.append(getattr(state, item))
        else:
            setting_list = "\n -".join(state.callback_properties() + list(override))
            raise MissingSettingError("This custom viewer is trying to use an "
                                      "unrecognized variable named %s\n. Valid "
                                      "variable names are\n -%s" %
                                      (item, setting_list))

    k = k or {}

    return func(*args, **k)


class MissingSettingError(KeyError):
    pass


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

        # Find ui elements
        ui = {}
        for key, value in list(attrs.items()):
            if key.startswith('_') or key in CustomViewer.__dict__:
                continue
            if not isinstance(value, (MethodType, FunctionType)):
                ui[key] = attrs.pop(key)

        attrs['ui'] = ui
        attrs.setdefault('name', name)

        # collect the user defined functions

        udfs = {}

        for nm, value in list(attrs.items()):
            dscr = CustomViewer.__dict__.get(nm, None)

            if isinstance(dscr, UserDefinedFunction):
                # remove them as class method
                # register them below instead
                udfs[nm] = attrs.pop(nm)

        result = type.__new__(cls, name, bases, attrs)
        result._custom_functions = {}

        # now wrap the custom user defined functions using the descriptors
        for k, v in udfs.items():
            # register UDF by mimicing the decorator syntax
            udf_decorator = getattr(result, k)
            udf_decorator(v)

        result._build_data_viewer()

        return result


class CustomSubsetState(SubsetState):
    """
    A SubsetState subclass that uses a CustomViewer's "select" function
    """

    def __init__(self, coordinator, roi):
        super(CustomSubsetState, self).__init__()
        self._coordinator = coordinator
        self._roi = roi

    def to_mask(self, data, view=None):
        return self._coordinator.select(layer=data, roi=self._roi, view=view)

    def copy(self):
        return CustomSubsetState(self._coordinator, self._roi)

    def __gluestate__(self, context):
        result = {}
        result['viewer'] = context.id(self._coordinator.viewer)
        result['roi'] = context.id(self._roi)
        return result

    @classmethod
    def __setgluestate__(cls, rec, context):
        roi = context.object(rec['roi'])
        subset_state = cls(None, roi)
        subset_state._viewer_rec = rec['viewer']
        return subset_state

    def __setgluestate_callback__(self, context):
        # When __setgluestate__ is created, the viewers might not yet be
        # deserialized, and these depend on the Data and Subsets existing so
        # we need to deserialize the viewer in a callback so it can be called
        # later on.
        viewer = context.object(self._viewer_rec)
        self._coordinator = viewer._coordinator
        self._viewer_rec = None


class BaseCustomOptionsWidget(QWidget):
    """
    Base class for the Qt widget which will be used to show the options.
    """

    _widgets = None

    def __init__(self, viewer_state=None, session=None):

        super(BaseCustomOptionsWidget, self).__init__()

        layout = QGridLayout()
        for row, (name, (prefix, viewer_cls)) in enumerate(self._widgets.items()):
            widget = viewer_cls()
            setattr(self, prefix + name, widget)
            layout.addWidget(QLabel(name.capitalize()), row, 0)
            layout.addWidget(widget, row, 1)
        if len(self._widgets) > 0:
            layout.setRowStretch(row + 1, 10)
        self.setLayout(layout)

        self.viewer_state = viewer_state
        self.session = session

        self._connections = autoconnect_callbacks_to_qt(self.viewer_state, self)


class CustomViewer(object, metaclass=CustomViewerMeta):

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

      - The name of a UI element (e.g. keywords passed to :func:`glue.custom_viewer`,
        or class-level variables in subclasses). The value assigned to this
        argument will be the current UI setting (e.g. booleans for checkboxes).
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

    # Label to give this widget in the GUI
    name = ''

    # Container to hold user descriptions of desired FormElements to create
    ui = {}

    # map, e.g., 'plot_data' -> user defined function - we also make sure we
    # override this in sub-classes in CustomViewerMeta
    _custom_functions = {}

    def __init__(self, viewer):
        self.viewer = viewer
        self.state = ViewerUserState()
        self.setup()

    @property
    def selections_enabled(self):
        return 'make_selector' in self._custom_functions or 'select' in self._custom_functions

    @classmethod
    def create_new_subclass(cls, name, **kwargs):
        """
        Convenience method to build a new CustomViewer subclass.

        This is used by the custom_viewer function.

        Parameters
        ----------
        name : str
            Name of the new viewer
        kwargs
            UI elements in the subclass
        """
        kwargs = kwargs.copy()
        kwargs['name'] = name
        # each subclass needs its own dict
        kwargs['_custom_functions'] = {}
        name = name.replace(' ', '')
        return CustomViewerMeta(name, (CustomViewer,), kwargs)

    @classmethod
    def _build_data_viewer(cls):
        """
        Build the DataViewer subclass for this viewer.
        """

        # At this point, the metaclass has put all the user options in a dict
        # called .ui, so we go over this dictionary and find the widgets and
        # callback properties for each of them.

        widgets = {}
        properties = {}

        for name in sorted(cls.ui):

            value = cls.ui[name]
            prefix, widget, property = FormElement.auto(value).ui_and_state()

            if widget is not None:
                widgets[name] = prefix, widget

            properties[name] = property

        options_cls = type(cls.__name__ + 'OptionsWidget',
                           (BaseCustomOptionsWidget,), {'_widgets': widgets})

        state_cls = type(cls.__name__ + 'ViewerState', (CustomMatplotlibViewerState,), properties)

        widget_dict = {'LABEL': cls.name,
                       'ui': cls.ui,
                       '_options_cls': options_cls,
                       '_state_cls': state_cls,
                       '_coordinator_cls': cls}

        viewer_cls = type(cls.__name__ + 'DataViewer',
                          (CustomMatplotlibDataViewer,),
                          widget_dict)

        cls._viewer_cls = viewer_cls
        qt_client.add(viewer_cls)

        # add new classes to module namespace
        # needed for proper state saving/restoring
        for c in [viewer_cls, cls]:
            mod = getmodule(ViewerUserState)
            w = getattr(mod, c.__name__, None)
            if w is not None:
                raise RuntimeError("Duplicate custom viewer detected %s" % c)
            setattr(mod, c.__name__, c)
            c.__module__ = mod.__name__

    @classmethod
    def _register_override_method(cls, name, func):
        """
        Register a new custom method like "plot_data"

        Users need not call this directly - it is called when a method is
        overridden or decorated
        """
        cls._custom_functions[name] = func

    def _build_subset_state(self, roi):

        if 'make_selector' in self._custom_functions:
            return self.make_selector(roi=roi)

        if 'select' in self._custom_functions:
            return CustomSubsetState(self, roi)

        raise RuntimeError("Selection not supported for this viewer.")

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

    def _call_udf(self, method_name, **kwargs):
        """
        Call a user-defined function stored in the _custom_functions dict

        Parameters
        ----------
        method_name : str
            The name of the user-defined method to setup a dispatch for
        use_cid : bool, optional
            Whether to pass component IDs to the user function instead of the
            data itself.
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
        artists and re-rendering the canvas as needed.
        """

        # get the custom function
        try:
            func = self._custom_functions[method_name]
        except KeyError:
            return []

        override = kwargs.copy()

        if 'layer' not in override and len(self.viewer.state.layers) > 0:
            override['layer'] = self.viewer.state.layers[0].layer

        if 'layer' in override:

            override.setdefault('style', override['layer'].style)

            # Dereference attributes
            for name, property in self.viewer.state.iter_callback_properties():
                value = getattr(self.viewer.state, name)
                if isinstance(value, ComponentID) or isinstance(property, FixedComponentIDProperty):
                    override[name] = AttributeWithInfo.from_layer(override['layer'], value, view=override.get('view', None))

        # add some extra information that the user might want
        override.setdefault('self', self)
        override.setdefault('axes', self.viewer.axes)
        override.setdefault('figure', self.viewer.axes.figure)
        override.setdefault('state', self.state)

        # call method, keep track of newly-added artists
        result = introspect_and_call(func, self.viewer.state, override)

        self.viewer.redraw()

        return result


class CustomLayerArtist(MatplotlibLayerArtist):
    """
    LayerArtist for simple custom viewers that use Matplotlib
    """

    _layer_state_cls = MatplotlibLayerState

    def __init__(self, coordinator, *args, **kwargs):
        super(CustomLayerArtist, self).__init__(*args, **kwargs)
        self._coordinator = coordinator
        self.state.add_global_callback(self.update)
        self._viewer_state.add_global_callback(self.update)

    def update(self, *args, **kwargs):

        if not self._visible:
            return

        self.clear()

        old = all_artists(self.axes.figure)

        if isinstance(self.state.layer, BaseData):
            a = self._coordinator.plot_data(layer=self.state.layer)
        else:
            a = self._coordinator.plot_subset(layer=self.state.layer, subset=self.state.layer)

        # if user explicitly returns the newly-created artists,
        # then use them. Otherwise, introspect to find the new artists
        if a is None:
            self.mpl_artists = list(new_artists(self.axes.figure, old))
        else:
            self.mpl_artists = as_list(a)

        for a in self.mpl_artists:
            a.set_zorder(self.state.zorder)


class CustomMatplotlibDataViewer(MatplotlibDataViewer):
    """
    Base Qt widget class for simple custom viewers that use Matplotlib
    """

    LABEL = ''
    tools = ['select:rectangle', 'select:polygon']

    _state_cls = None
    _options_cls = None
    _layer_style_viewer_cls = None
    _data_artist_cls = CustomLayerArtist
    _subset_artist_cls = CustomLayerArtist

    _coordinator_cls = None

    def __init__(self, session, parent=None, **kwargs):
        super(CustomMatplotlibDataViewer, self).__init__(session, parent, **kwargs)
        self._coordinator = self._coordinator_cls(self)
        self.state.add_global_callback(self._on_state_change)
        self._on_state_change()

    def _on_state_change(self, *args, **kwargs):
        self._coordinator.settings_changed()

    def get_layer_artist(self, cls, layer=None, layer_state=None):
        return cls(self._coordinator, self.axes, self.state, layer=layer, layer_state=layer_state)

    @defer_draw
    def apply_roi(self, roi):

        # Force redraw to get rid of ROI. We do this because applying the
        # subset state below might end up not having an effect on the viewer,
        # for example there may not be any layers, or the active subset may not
        # be one of the layers. So we just explicitly redraw here to make sure
        # a redraw will happen after this method is called.
        self.redraw()

        if len(self.layers) == 0:
            return

        subset_state = self._coordinator._build_subset_state(roi=roi)
        self.apply_subset_state(subset_state)


class CustomMatplotlibViewerState(MatplotlibDataViewerState):

    def __init__(self, *args, **kwargs):
        super(CustomMatplotlibViewerState, self).__init__(*args)
        self._cid_helpers = []
        for name, property in self.iter_callback_properties():
            if isinstance(property, DynamicComponentIDProperty):
                self._cid_helpers.append(ComponentIDComboHelper(self, name))
        self.add_callback('layers', self._on_layer_change)
        self.update_from_dict(kwargs)

    def _on_layer_change(self, *args):
        for helper in self._cid_helpers:
            helper.set_multiple_data(self.layers_data)
