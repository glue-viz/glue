from inspect import getargspec

from PyQt4.QtGui import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit
from PyQt4.QtGui import QSpacerItem, QSizePolicy

from PyQt4.QtCore import Qt

from .ui.link_equation import Ui_LinkEquation
from .. import core

def function_label(function):
    """ Format a function signature as a string """
    name = function.__name__
    args = getargspec(function)[0]
    label = "output = %s(%s)" % (name, ', '.join(args))
    return label


class ArgumentWidget(QWidget):
    def __init__(self, argument, parent=None):
        super(ArgumentWidget, self).__init__(parent)
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(1, 0, 1, 1)
        self.setLayout(self.layout)
        label = QLabel(argument)
        self.component_id = None
        self.layout.addWidget(label)
        self.editor = QLineEdit()
        self.editor.setReadOnly(True)
        self.editor.setPlaceholderText("Drag a component from above")
        self.layout.addWidget(self.editor)
        self.setAcceptDrops(True)

    def clear(self):
        self.editor.clear()

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat('application/py_instance'):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        obj = event.mimeData().data('application/py_instance')
        if not isinstance(obj, core.data.ComponentID):
            return
        self.component_id = obj
        self.editor.setText(str(obj))


class LinkEquation(QWidget):
    """ Interactively define ComponentLinks from existing functions

    This widget inspects the calling signatures of input functions,
    and presents the user with an interface for assigning componentIDs
    to the input and output arguments. It also generates ComponentLinks
    from this information.

    ComponentIDs are assigned to arguments via drag and drop. This
    widget is used within the LinkEditor dialog

    Usage::

       widget = LinkEquation()
       widget.setup([list_of_functions])

    """

    def __init__(self, parent=None):
        super(LinkEquation, self).__init__(parent)
        self._functions = None
        self._argument_widgets = []
        self.spacer = None
        self._output_widget = ArgumentWidget("")
        self._ui = Ui_LinkEquation()

        self._init_widgets()
        self._connect()

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

    def setup(self, functions):
        """ Set up widgets for assigning to a list of functions

        :param functions: functions to interact with
        :type functions: list of function objects
        """
        self._functions = dict(((f.__name__, f) for f in functions))
        self._populate_function_combo()

    @property
    def add_button(self):
        return self._ui.addButton

    @property
    def two_way(self):
        """ Returns true if the link is two-way

        A link is two way if the input and output arguments can be
        flipped. If the link is two-way, two links will be created
        """
        return self._ui.two_way.checkState() == Qt.Checked

    @property
    def signature(self):
        """ Returns the ComponentIDs assigned to the input and output arguments

        :rtype: tuple of (input, output). Input is a list of ComponentIDs.
                output is a ComponentID
        """
        inp = [a.component_id for a in self._argument_widgets]
        out = self._output_widget.component_id
        return inp, out

    @property
    def _function(self):
        """ The currently-selected function """
        fname = str(self._ui.function.currentText())
        func = self._functions[fname]
        return func

    def links(self):
        """ Create ComponentLinks from the state of the widget

        :rtype: list of ComponentLinks that can be created.

        If no links can be created (e.g. because of missing input),
        the empty list is returned
        """
        inp, out = self.signature
        if not inp or not out:
            return []
        using = self._function

        link = core.component_link.ComponentLink(inp, out, using)
        if self.two_way:
            link2 = core.component_link.ComponentLink([out], inp[0], using)
            return [link, link2]

        return [link]

    def _update_add_enabled(self):
        state = True
        for a in self._argument_widgets:
            state = state and a.component_id is not None
        state = state and self._output_widget.component_id is not None
        self._ui.addButton.setEnabled(state)

    def _update_twoway_enabled(self):
        self._ui.two_way.setEnabled(len(self._argument_widgets) == 1)
        if not self._ui.two_way.isEnabled():
            self._ui.two_way.setCheckState(Qt.Unchecked)

    def _connect(self):
        signal = self._ui.function.currentIndexChanged
        signal.connect(self._setup_editor)
        signal.connect(self._update_add_enabled)
        signal.connect(self._update_twoway_enabled)
        self._ui.addButton.pressed.connect(self._clear_inputs)
        self._output_widget.editor.textChanged.connect(
            self._update_add_enabled)

    def _clear_inputs(self):
        for w in self._argument_widgets:
            w.clear()
        self._output_widget.clear()

    def _setup_editor(self):
        """ Prepare the widget for the active function.
        """
        func = self._function
        args = getargspec(func)[0]
        label = function_label(func)
        self._ui.function_signature.setText(label)

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
        layout.removeItem(self.spacer)
        self._argument_widgets = []

    def _populate_function_combo(self):
        """ Add name of functions to function combo box """
        self._ui.function.clear()
        for f in self._functions:
            self._ui.function.addItem(f)


def main(): # pragma: no cover
    import glue
    import numpy as np
    from component_selector import ComponentSelector
    from PyQt4.QtGui import QApplication

    app = QApplication([''])

    d = glue.Data(label = 'd1')
    d2 = glue.Data(label = 'd2')
    c1 = glue.Component(np.array([1, 2, 3]))
    c2 = glue.Component(np.array([1, 2, 3]))
    c3 = glue.Component(np.array([1, 2, 3]))
    d.add_component(c1, 'a')
    d.add_component(c2, 'b')
    d2.add_component(c3, 'c')
    dc = glue.DataCollection()
    dc.append(d)
    dc.append(d2)

    def f(a, b, c):
        pass

    def g(h, i):
        pass

    def h(j, k=None):
        pass

    w = LinkEquation()
    w.setup([f, g, h])

    cs = ComponentSelector(dc, label = "Data 1")
    w.show()
    cs.show()
    app.exec_()

if __name__ == "__main__":
    main()