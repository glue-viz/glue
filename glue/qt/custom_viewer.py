"""
This module provides utilities for creating custom data viewers. The
goal of this module is to make it easy for users to make new
data viewers by focusing on matplotlib visualization logic,
and not UI or event processing logic.

The end user typically interacts with this code via
:func:`glue.custom_viewer`
"""
from ..clients.layer_artist import LayerArtist
from ..clients.dendro_client import GenericMplClient
from ..core import Data
from ..core.edit_subset_mode import EditSubsetMode
from ..core.util import nonpartial, as_list
from .. import core

from .widgets.data_viewer import DataViewer
from .widget_properties import CurrentComboProperty
from ..external.qt import QtGui
from ..external.qt.QtCore import Qt
from .widgets import MplWidget
from .glue_toolbar import GlueToolbar
from .mouse_mode import LassoMode, RectangleMode

CUSTOM_WIDGETS = []


def noop(*a, **k):
    return []


class CustomViewerFactory(object):

    """
    Decorator class to build custom data viewers.

    The public methods of this class are decorators, that
    wrap custom viewer methods.
    """

    def __init__(self, name, **kwargs):
        """

        :param name: The name of the custom viewer
        :type name: str

        Extra kwargs are used to specify the User interface
        """
        lbl = name.replace(' ', '')
        artist_cls = type('%sLayerArtist' % lbl, (AutoLayerArtist,), {})
        client_cls = type('%sClient' % lbl, (AutoClient,), {'artist_cls': artist_cls})
        widget_dict = {'client_cls': client_cls, 'LABEL': name, 'ui': kwargs}
        widget_cls = type('%sWidget' % lbl, (AutoWidget,), widget_dict)
        self._artist_cls = artist_cls
        self._client_cls = client_cls
        self._widget_cls = widget_cls
        CUSTOM_WIDGETS.append(widget_cls)

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
        return func


class AutoLayerArtist(LayerArtist):
    plot_data = noop
    plot_subset = noop

    def __init__(self, layer, axes, settings):
        super(AutoLayerArtist, self).__init__(layer, axes)
        self.settings = settings

    def update(self, view=None):

        kwargs = dict(style=self._layer.style)
        for k, v in self.settings.items():
            kwargs[k] = v.value(self._layer)

        self.clear()

        if isinstance(self._layer, Data):
            artists = self.plot_data(self._axes, **kwargs)
        else:
            artists = self.plot_subset(self._axes, **kwargs)

        for a in as_list(artists):
            a.set_zorder(self.zorder)

        self.artists = as_list(artists)


class AutoClient(GenericMplClient):
    artist_cls = AutoLayerArtist
    setup_func = noop
    make_selector = noop

    def __init__(self, *args, **kwargs):
        self.settings = kwargs.pop('settings', {})
        super(AutoClient, self).__init__(*args, **kwargs)
        self.setup_func(self.axes)

    def new_layer_artist(self, layer):
        return self.artist_cls(layer, self.axes, self.settings)

    def apply_roi(self, roi):
        if len(self.artists) > 0:
            focus = self.artists[0].layer.data
        elif len(self.collect) > 0:
            focus = self.collect[0]
        else:
            return

        s = self.make_selector(roi,
                               **FormElement.dereference(self.settings))
        if s:
            EditSubsetMode().update(self.collect, s, focus_data=focus)

    def _update_layer(self, layer):
        for artist in self.artists[layer]:
            artist.update()

        self._redraw()


class AutoWidget(DataViewer):
    LABEL = 'Auto'
    client_cls = AutoClient
    update_settings = noop
    make_selector = noop

    def __init__(self, session, parent=None):
        super(AutoWidget, self).__init__(session, parent)
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
        result.setLayout(layout)
        for k, v in type(self).ui.items():
            w = FormElement.auto(v)
            w.container = self._container
            w.add_callback(self.settings_changed)
            self._settings[k] = w
            if w.ui is not None:
                layout.addRow(k, w.ui)

        return result

    def settings_changed(self):
        for a in self._update_artists:
            try:
                a.remove()
            except ValueError:  # already removed
                pass
        self._update_artists = self.update_settings(self.client.axes,
                                                    **self.settings())
        self.client._redraw()

    def make_toolbar(self):
        result = GlueToolbar(self.central_widget.canvas, self, name=self.LABEL)
        for mode in self._mouse_modes():
            result.add_mode(mode)
        self.addToolBar(result)
        return result

    def _mouse_modes(self):
        axes = self.client.axes

        def apply_mode(mode):
            self.client.apply_roi(mode.roi())

        return [RectangleMode(axes, roi_callback=apply_mode),
                LassoMode(axes, roi_callback=apply_mode)]

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
        super(AutoWidget, self).register_to_hub(hub)
        self.client.register_to_hub(hub)
        for w in self._settings.values():
            w.register_to_hub(hub)

    def unregister(self, hub):
        super(AutoWidget, self).unregister(hub)
        hub.unsubscribe_all(self.client)
        hub.unsubscribe_all(self)
        for w in self._settings.values():
            hub.unsubscribe_all(w)


class FormElement(object):

    def __init__(self, params):
        self.params = params
        self._callbacks = []
        self.ui = self._build_ui()
        self.container = None  # layer container

    def _build_ui(self):
        raise NotImplementedError()

    def value(self, data=None):
        raise NotImplementedError()

    def changed(self):
        for cb in self._callbacks:
            cb()

    def add_callback(self, cb):
        self._callbacks.append(cb)

    @classmethod
    def recognizes(cls, params):
        raise NotImplementedError

    @staticmethod
    def auto(params):
        for cls in FormElement.__subclasses__():
            if cls.recognizes(params):
                return cls(params)
        raise ValueError("Unrecognzied UI Component: %s" % params)

    @staticmethod
    def dereference(elements, layer=None):
        return dict((k, v.value(layer)) for k, v in elements.items())

    def register_to_hub(self, hub):
        pass

    def add_data(self, data):
        pass


class NumberElement(FormElement):

    @classmethod
    def recognizes(cls, params):
        return isinstance(params, tuple)

    def _build_ui(self):
        w = QtGui.QSlider()
        w.setMinimum(self.params[0])
        w.setMaximum(self.params[1])
        w.setValue((self.params[0] + self.params[1]) / 2)
        w.setOrientation(Qt.Horizontal)
        w.valueChanged.connect(nonpartial(self.changed))
        return w

    def value(self, data=None):
        return self.ui.value()


class BoolElement(FormElement):

    @classmethod
    def recognizes(cls, params):
        return isinstance(params, bool)

    def _build_ui(self):
        w = QtGui.QCheckBox()
        w.setChecked(self.params)
        w.toggled.connect(nonpartial(self.changed))
        return w

    def value(self, data=None):
        return self.ui.isChecked()


class FixedComponent(FormElement):

    @classmethod
    def recognizes(cls, params):
        try:
            return params.startswith('att(')
        except AttributeError:
            return False

    def _build_ui(self):
        pass

    def value(self, data=None):
        if data is not None:
            return data[self.params.split('(')[-1][:-1]]


class ComponenentElement(FormElement, core.hub.HubListener):
    _component = CurrentComboProperty('ui')

    @classmethod
    def recognizes(cls, params):
        return params == 'att'

    def _build_ui(self):
        result = QtGui.QComboBox()
        result.currentIndexChanged.connect(nonpartial(self.changed))
        return result

    def value(self, data=None):
        cid = self._component
        if data is None or cid is None:
            return cid
        return data[cid]

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

    @classmethod
    def recognizes(cls, params):
        return isinstance(params, (list, dict))

    def _build_ui(self):
        w = QtGui.QComboBox()
        for p in self.params:
            w.addItem(p)

        if isinstance(self.params, list):
            self.params = dict((p, p) for p in self.params)

        w.currentIndexChanged.connect(nonpartial(self.changed))
        return w

    def value(self, data=None):
        return self.params[self.ui.currentText()]
