from __future__ import print_function, division

from glue.external import six
from glue.external.echo import CallbackProperty, SelectionCallbackProperty

from qtpy.QtWidgets import QSlider, QLineEdit, QComboBox, QWidget, QLabel, QHBoxLayout, QCheckBox
from qtpy.QtCore import Qt

__all__ = ["FormElement", "NumberElement", "LabeledSlider",
           "BoolElement", "ChoiceElement", "FixedComponent"]


class LabeledSlider(QWidget):

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
        self._slider = QSlider()
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
        self._lbl = QLabel(str(self.value()))
        self._l = QHBoxLayout()
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


class FormElement(object):

    """
    Base class for user-defined settings in a custom widget.

    Each form element has a value() and a widget. Subclasses
    must override _build_ui, value, and recognizes. They
    may override register_to_hub and add_data.
    """

    def __init__(self, params):
        self.params = params

    @classmethod
    def recognizes(cls, params):
        """
        Returns whether or not a shorthand "params" object
        can be passed to __init__ to construct an element
        """
        raise NotImplementedError

    def ui_and_state(self):
        """
        Build and return a widget to represent this setting.
        """
        raise NotImplementedError()

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


class NumberElement(FormElement):

    """
    A form element representing a number

    The shorthand is a tuple of 2 or 3 numbers:
    (min, max) or (min, max default)::

        e = FormElement.auto((0., 1.))
    """

    @classmethod
    def recognizes(cls, params):
        try:
            if len(params) not in [2, 3]:
                return False
            return all(isinstance(p, six.integer_types + (float,)) for p in params)
        except TypeError:
            return False

    def ui_and_state(self):
        # widget = LabeledSlider(*self.params[:2])
        default = self.params[2] if len(self.params) == 3 else self.params[0]
        return 'value_', QSlider, CallbackProperty(default)


class TextBoxElement(FormElement):
    """
    A form element representing a generic textbox

    The shorthand is any string starting with an _.::

        e = FormElement.auto("_default")

    Everything after the underscore is taken as the default value.
    """

    @classmethod
    def recognizes(cls, params):
        try:
            if isinstance(params, str) & params.startswith('_'):
                return True
        except AttributeError:
            return None

    def ui_and_state(self):
        default = self.params[1:]
        return 'text_', QLineEdit, CallbackProperty(default)


class FloatElement(FormElement):
    """
    A form element representing a generic number box.

    The shorthand is any number::

        e = FormElement.auto(2)

    The number itself is taken as the default value.
    """

    @classmethod
    def recognizes(cls, params):
        return isinstance(params, (int, float)) and not isinstance(params, bool)

    def ui_and_state(self):
        default = self.params
        return 'valuetext_', QLineEdit, CallbackProperty(default)


class BoolElement(FormElement):

    """
    A checkbox representing a boolean setting

    The shorthand notation is True or False::

        e = FormElement.auto(False)
    """

    @classmethod
    def recognizes(cls, params):
        return isinstance(params, bool)

    def ui_and_state(self):
        default = self.params
        return 'bool_', QCheckBox, CallbackProperty(default)


class ChoiceElement(FormElement):

    """
    A dropdown selector to choose between a set of items

    Shorthand notation is a sequence of strings or a dict::

        e = FormElement.auto({'a':1, 'b':2})
        e = FormElement.auto(['a', 'b', 'c'])
    """

    @classmethod
    def recognizes(cls, params):
        if isinstance(params, six.string_types):
            return False
        try:
            return all(isinstance(p, six.string_types) for p in params)
        except TypeError:
            return False

    def ui_and_state(self):
        if isinstance(self.params, list):
            choices = dict((p, p) for p in self.params)
        else:
            choices = self.params
        property = SelectionCallbackProperty()
        # property = SelectionCallbackProperty(choices=choices)
        return 'combosel_', QComboBox, property


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

    def ui_and_state(self):
        return None, None, CallbackProperty(self.params)
#
#
# class ComponenentElement(FormElement, core.hub.HubListener):
#
#     """
#     A dropdown selector to choose a component
#
#     The shorthand notation is 'att'::
#
#         e = FormElement.auto('att')
#     """
#     _component = CurrentComboProperty('ui')
#
#     @property
#     def state(self):
#         return self._component
#
#     @state.setter
#     def state(self, value):
#         self._update_components()
#         if value is None:
#             return
#         self._component = value
#
#     @classmethod
#     def recognizes(cls, params):
#         return params == 'att'
#
#     def _build_ui(self):
#         result = QComboBox()
#         result.currentIndexChanged.connect(nonpartial(self.changed))
#         return result
#
#     def value(self, layer=None, view=None):
#         cid = self._component
#         if layer is None or cid is None:
#             return AttributeInfo.make(cid, [], None)
#         return AttributeInfo.from_layer(layer, cid, view)
#
#     def _list_components(self):
#         """
#         Determine which components to list.
#
#         This can be overridden by subclassing to limit which components are
#         visible to the user.
#
#         """
#         comps = list(set([c for l in self.container.layers
#                           for c in (l.data.main_components + l.data.derived_components)]))
#         comps = sorted(comps, key=lambda x: x.label)
#         return comps
#
#     def _update_components(self):
#         combo = self.ui
#         old = self._component
#
#         combo.blockSignals(True)
#         combo.clear()
#
#         comps = self._list_components()
#         for c in comps:
#             combo.addItem(c.label, userData=UserDataWrapper(c))
#
#         try:
#             combo.setCurrentIndex(comps.index(old))
#         except ValueError:
#             combo.setCurrentIndex(0)
#
#         combo.blockSignals(False)
#
#     def register_to_hub(self, hub):
#         hub.subscribe(self, core.message.ComponentsChangedMessage,
#                       nonpartial(self._update_components))
#
#     def add_data(self, data):
#         self._update_components()
