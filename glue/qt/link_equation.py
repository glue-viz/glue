from inspect import getargspec

from ..external.qt.QtGui import (QWidget, QHBoxLayout, QVBoxLayout,
                                 QLabel, QLineEdit)

from ..external.qt.QtGui import QSpacerItem, QSizePolicy
from ..external.qt import QT_API, QT_API_PYSIDE

from .ui.link_equation import Ui_LinkEquation
from .. import core
from ..core.odict import OrderedDict


def function_label(function):
    """ Provide a label for a function

    :param function: A member from the glue.config.link_function registry
    """
    name = function.function.__name__
    args = getargspec(function.function)[0]
    args = ', '.join(args)
    output = function.output_labels
    output = ', '.join(output)
    label = "Link from %s to %s" % (args, output)
    return label


def helper_label(helper):
    """ Provide a label for a link helper

    :param helper: A member from the glue.config.link_helper registry
    """
    return helper.info


class ArgumentWidget(QWidget):
    def __init__(self, argument, parent=None):
        super(ArgumentWidget, self).__init__(parent)
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(1, 0, 1, 1)
        self.setLayout(self.layout)
        label = QLabel(argument)
        self._label = label
        self._component_id = None
        self.layout.addWidget(label)
        self.editor = QLineEdit()
        self.editor.setReadOnly(True)
        try:
            self.editor.setPlaceholderText("Drag a component from above")
        except AttributeError:  # feature added in Qt 4.7
            pass
        self.layout.addWidget(self.editor)
        self.setAcceptDrops(True)

    @property
    def component_id(self):
        return self._component_id

    @component_id.setter
    def component_id(self, cid):
        self._component_id = cid
        self.editor.setText(str(cid))

    @property
    def label(self):
        return self._label.text()

    @label.setter
    def label(self, label):
        self._label.setText(label)

    @property
    def editor_text(self):
        return self.editor.text()

    def clear(self):
        self.component_id = None
        self.editor.clear()

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat('application/py_instance'):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        obj = event.mimeData().data('application/py_instance')
        if isinstance(obj, list):
            obj = obj[0]
        if not isinstance(obj, core.data.ComponentID):
            event.ignore()
            return
        self.component_id = obj
        event.accept()


class LinkEquation(QWidget):
    """ Interactively define ComponentLinks from existing functions

    This widget inspects the calling signatures of helper functions,
    and presents the user with an interface for assigning componentIDs
    to the input and output arguments. It also generates ComponentLinks
    from this information.

    ComponentIDs are assigned to arguments via drag and drop. This
    widget is used within the LinkEditor dialog

    Usage::

       widget = LinkEquation()
    """

    def __init__(self, parent=None):
        super(LinkEquation, self).__init__(parent)
        from ..config import link_function, link_helper
        #map of function/helper name -> function/helper tuple
        f = [f for f in link_function.members if len(f.output_labels) == 1]
        self._functions = OrderedDict((l[0].__name__, l) for l in
                                      f + link_helper.members)
        self._argument_widgets = []
        self.spacer = None
        self._output_widget = ArgumentWidget("")
        self._ui = Ui_LinkEquation()

        self._init_widgets()
        self._populate_function_combo()
        self._connect()
        self._setup_editor()

    def set_result_visible(self, state):
        self._ui.output_canvas.setVisible(state)
        self._ui.output_label.setVisible(state)

    def is_helper(self):
        return self.function is not None and \
            type(self.function).__name__ == 'LinkHelper'

    def is_function(self):
        return self.function is not None and \
            type(self.function).__name__ == 'LinkFunction'

    def _init_widgets(self):
        self._ui.setupUi(self)
        layout = QVBoxLayout()
        layout.setSpacing(1)
        self._ui.input_canvas.setLayout(layout)
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 0, 1, 1)
        self._ui.output_canvas.setLayout(layout)
        layout.addWidget(self._output_widget)
        spacer = QSpacerItem(5, 5, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer)

    @property
    def add_button(self):
        return self._ui.addButton

    @property
    def signature(self):
        """ Returns the ComponentIDs assigned to the input and output arguments

        :rtype: tuple of (input, output). Input is a list of ComponentIDs.
                output is a ComponentID
        """
        inp = [a.component_id for a in self._argument_widgets]
        out = self._output_widget.component_id
        return inp, out

    @signature.setter
    def signature(self, inout):
        inp, out = inout
        for i, a in zip(inp, self._argument_widgets):
            a.component_id = i
        self._output_widget.component_id = out

    @property
    def function(self):
        """ The currently-selected function

        :rtype: A function or helper tuple
        """
        fname = str(self._ui.function.currentText())
        func = self._functions[fname]
        return func

    @function.setter
    def function(self, val):
        name = val[0].__name__
        pos = self._ui.function.findText(name)
        if pos < 0:
            raise KeyError("No function or helper found %s" % [val])
        self._ui.function.setCurrentIndex(pos)

    def links(self):
        """ Create ComponentLinks from the state of the widget

        :rtype: list of ComponentLinks that can be created.

        If no links can be created (e.g. because of missing input),
        the empty list is returned
        """
        inp, out = self.signature
        if self.is_function():
            using = self.function.function
            if not all(inp) or not out:
                return []
            link = core.component_link.ComponentLink(inp, out, using)
            return [link]
        if self.is_helper():
            helper = self.function.helper
            if not all(inp):
                return []
            return helper(*inp)

    def _update_add_enabled(self):
        state = True
        for a in self._argument_widgets:
            state = state and a.component_id is not None
        if self.is_function():
            state = state and self._output_widget.component_id is not None
        self._ui.addButton.setEnabled(state)

    def _connect(self):
        signal = self._ui.function.currentIndexChanged
        signal.connect(self._setup_editor)
        signal.connect(self._update_add_enabled)
        self._output_widget.editor.textChanged.connect(
            self._update_add_enabled)

    def clear_inputs(self):
        for w in self._argument_widgets:
            w.clear()
        self._output_widget.clear()

    def _setup_editor(self):
        if self.is_function():
            self._setup_editor_function()
        else:
            self._setup_editor_helper()

    def _setup_editor_function(self):
        """ Prepare the widget for the active function."""
        assert self.is_function()
        self.set_result_visible(True)
        func = self.function.function
        args = getargspec(func)[0]
        label = function_label(self.function)
        self._ui.info.setText(label)
        self._output_widget.label = self.function.output_labels[0]
        self._clear_input_canvas()
        for a in args:
            self._add_argument_widget(a)

        self.spacer = QSpacerItem(5, 5, QSizePolicy.Minimum,
                                  QSizePolicy.Expanding)
        self._ui.input_canvas.layout().addItem(self.spacer)

    def _setup_editor_helper(self):
        """Setup the editor for the selected link helper"""
        assert self.is_helper()
        self.set_result_visible(False)
        label = helper_label(self.function)
        args = self.function.input_labels
        self._ui.info.setText(label)

        self._clear_input_canvas()
        for a in args:
            self._add_argument_widget(a)

        self.spacer = QSpacerItem(5, 5, QSizePolicy.Minimum,
                                  QSizePolicy.Expanding)
        self._ui.input_canvas.layout().addItem(self.spacer)

    def _add_argument_widget(self, argument):
        """ Create and add a single argument widget to the input canvas
        :param arguement: The argument name (string)
        """
        widget = ArgumentWidget(argument)
        widget.editor.textChanged.connect(self._update_add_enabled)
        self._ui.input_canvas.layout().addWidget(widget)
        self._argument_widgets.append(widget)

    def _clear_input_canvas(self):
        """ Remove all widgets from the input canvas """
        layout = self._ui.input_canvas.layout()
        for a in self._argument_widgets:
            layout.removeWidget(a)
            a.close()

        if QT_API != QT_API_PYSIDE:
            #XXX PySide crashing here
            layout.removeItem(self.spacer)

        self._argument_widgets = []

    def _populate_function_combo(self):
        """ Add name of functions to function combo box """
        self._ui.function.clear()
        for f in self._functions:
            self._ui.function.addItem(f)
