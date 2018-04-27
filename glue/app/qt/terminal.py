"""
A GUI Ipython terminal window which can interact
with Glue. Based on code from

  https://stackoverflow.com/a/9796491/1332492

and

  https://stackoverflow.com/a/11525205/1332492

Usage:
   new_widget = glue_terminal(**kwargs)
"""

from __future__ import absolute_import, division, print_function

# must import these first, to set up Qt properly
from qtpy import QtWidgets

from IPython import get_ipython

from ipykernel.inprocess.ipkernel import InProcessInteractiveShell
from ipykernel.connect import get_connection_file

from qtconsole.inprocess import QtInProcessKernelManager
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.client import QtKernelClient

from glue.app.qt.mdi_area import GlueMdiSubWindow
from glue.utils import as_variable_name
from glue.utils.qt import get_qapp

kernel_manager = None
kernel_client = None


class IPythonTerminalError(Exception):
    pass


def start_in_process_kernel():

    global kernel_manager, kernel_client

    kernel_manager = QtInProcessKernelManager()
    kernel_manager.start_kernel()

    kernel_client = kernel_manager.client()
    kernel_client.start_channels()


def in_process_console(console_class=RichJupyterWidget, **kwargs):
    """
    Create a console widget, connected to an in-process Kernel

    Keyword arguments will be added to the namespace of the shell.

    Parameters
    ----------
    console_class : `type`
        The class of the console widget to create
    """

    global kernel_manager, kernel_client

    if kernel_manager is None:
        start_in_process_kernel()

    app = get_qapp()

    def stop():
        kernel_client.stop_channels()
        kernel_manager.shutdown_kernel()
        app.exit()

    control = console_class()
    control._display_banner = False
    control.kernel_manager = kernel_manager
    control.kernel_client = kernel_client
    control.exit_requested.connect(stop)
    control.shell = kernel_manager.kernel.shell
    control.shell.user_ns.update(**kwargs)
    control.setWindowTitle('IPython Terminal - type howto() for instructions')

    return control


def connected_console(console_class=RichJupyterWidget, **kwargs):
    """
    Create a console widget, connected to another kernel running in the current
    process.

    This is used for instance if glue has been started from IPython.

    Keyword arguments will be added to the namespace of the shell.

    Parameters
    ----------
    console_class : `type`
        The class of the console widget to create
    """

    shell = get_ipython()

    if shell is None:
        raise RuntimeError("There is no IPython kernel in this process")

    client = QtKernelClient(connection_file=get_connection_file())
    client.load_connection_file()
    client.start_channels()

    control = console_class()
    control.kernel_client = client
    control.shell = shell
    control.shell.user_ns.update(**kwargs)

    return control


glue_banner = """
This is the built-in IPython terminal. You can type any valid Python code here, and you also have access to the following pre-defined variables:

  * data_collection (aliased to dc)
  * application
  * hub

In addition, you can drag and drop any dataset or subset onto the terminal to create a new variable, and you will be prompted for a name.
"""


def howto():
    print(glue_banner.strip())


class DragAndDropTerminal(RichJupyterWidget):

    def __init__(self, **kwargs):
        super(DragAndDropTerminal, self).__init__(**kwargs)
        self.setAcceptDrops(True)
        self.shell = None

    def mdi_wrap(self):
        sub = GlueMdiSubWindow()
        sub.setWidget(self)
        self.destroyed.connect(sub.close)
        sub.resize(self.size())
        self._mdi_wrapper = sub
        return sub

    @property
    def namespace(self):
        return self.shell.user_ns if self.shell is not None else None

    def dragEnterEvent(self, event):
        fmt = 'application/py_instance'
        if self.shell is not None and event.mimeData().hasFormat(fmt):
            event.accept()
        else:
            event.ignore()

    def update_namespace(self, kwargs):
        if self.shell is not None:
            self.shell.push(kwargs)

    def dropEvent(self, event):
        obj = event.mimeData().data('application/py_instance')

        try:
            lbl = obj[0].label
        except (IndexError, AttributeError):
            lbl = 'x'
        lbl = as_variable_name(lbl)
        var, ok = QtWidgets.QInputDialog.getText(self, "Choose a variable name",
                                                 "Choose a variable name", text=lbl)
        if ok:
            # unpack single-item lists for convenience
            if isinstance(obj, list) and len(obj) == 1:
                obj = obj[0]

            var = {as_variable_name(str(var)): obj}
            self.update_namespace(var)
            event.accept()
        else:
            event.ignore()

    def clear_ns(self, names):
        if self.shell is not None:
            for name in names:
                self.shell.user_ns.pop(name)


FROM_TERMINAL_MESSAGE = """
Due to a limitation in IPython, the IPython terminal in glue does not work well
or at all if glue was started from an IPython terminal, or a Jupyter qtconsole
or notebook. However, you can use the original Python session you started glue
from instead to interact with glue objects. For example, if you started the
application with:

    In [2]: app = qglue(...)

or

    In [2]: app = GlueApplication(...)

you can then access e.g. the viewers with app.viewers, the data collection with
app.data_collection, and so on. If the IPython shell hangs and doesn't allow
you to type anything else until glue is closed, make sure you type:

    In [1]: %gui qt

before starting glue to avoid this kind of issue.
""".strip()


def glue_terminal(**kwargs):
    """
    Return a qt widget which embed an IPython interpreter.

    Keyword arguments will be added to the namespace of the shell.
    """

    kwargs['howto'] = howto
    shell = get_ipython()

    if shell is None or isinstance(shell, InProcessInteractiveShell):
        return in_process_console(console_class=DragAndDropTerminal, **kwargs)
    else:
        raise IPythonTerminalError(FROM_TERMINAL_MESSAGE)

    # TODO: if glue is launched from a qtconsole or a notebook, we should in
    # principle be able to use the connected_console function above, but this
    # doesn't behave quite right and requires further investigation.

    # Note however that if we do this, then we need to avoid polluting the
    # namespace of the original IPython console, as described in this issue:
    # https://github.com/glue-viz/glue/issues/1209
