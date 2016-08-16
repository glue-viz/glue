"""
A GUI Ipython terminal window which can interact
with Glue. Based on code from

http://stackoverflow.com/a/9796491/1332492
and
http://stackoverflow.com/a/11525205/1332492

Usage:
   new_widget = glue_terminal(**kwargs)

Implementation Note:

Since v1.0dev, IPython implements embeddable in-process terminal widgets.
This functionality doesn't exist in v0.12 and v0.13 -- this module provides
a fallback implmentation for older IPython versions
"""

from __future__ import absolute_import, division, print_function

import sys
import atexit
from contextlib import contextmanager
from distutils.version import LooseVersion

# must import these first, to set up Qt properly
from qtpy import QtCore, QtWidgets

import IPython
from IPython.core.usage import default_banner
from zmq import ZMQError
from zmq.eventloop import ioloop
from zmq.eventloop.zmqstream import ZMQStream

from glue.version import __version__

IPYTHON_VERSION = LooseVersion(IPython.__version__)

if IPYTHON_VERSION >= LooseVersion('4'):

    from IPython import get_ipython

    from traitlets import TraitError

    from ipykernel import find_connection_file
    from ipykernel.kernelbase import Kernel
    from ipykernel.kernelapp import IPKernelApp
    from ipykernel.iostream import OutStream
    from ipykernel.inprocess.ipkernel import InProcessInteractiveShell
    from ipykernel.connect import get_connection_file

    from qtconsole.client import QtKernelClient
    from qtconsole.manager import QtKernelManager
    from qtconsole.inprocess import QtInProcessKernelManager
    from qtconsole.rich_jupyter_widget import RichJupyterWidget as RichIPythonWidget

else:

    from IPython.utils.traitlets import TraitError
    from IPython.lib.kernel import find_connection_file

    from IPython import get_ipython

    from IPython.kernel.zmq.ipkernel import Kernel
    from IPython.kernel.zmq.kernelapp import IPKernelApp
    from IPython.kernel.zmq.iostream import OutStream
    from IPython.kernel.inprocess.ipkernel import InProcessInteractiveShell
    from IPython.kernel.connect import get_connection_file

    from IPython.qt.client import QtKernelClient
    from IPython.qt.manager import QtKernelManager
    from IPython.qt.inprocess import QtInProcessKernelManager
    from IPython.qt.console.rich_ipython_widget import RichIPythonWidget

from glue.app.qt.mdi_area import GlueMdiSubWindow
from glue.utils import as_variable_name


def in_process_console(console_class=RichIPythonWidget, **kwargs):
    """Create a console widget, connected to an in-process Kernel

    This only works on IPython v 0.13 and above

    Parameters
    ----------
    console_class : The class of the console widget to create
    kwargs : Extra variables to put into the namespace
    """

    km = QtInProcessKernelManager()
    km.start_kernel()

    kernel = km.kernel
    kernel.gui = 'qt4'

    client = km.client()
    client.start_channels()

    control = console_class()
    control.kernel_manager = km
    control.kernel_client = client
    control.shell = kernel.shell
    control.shell.user_ns.update(**kwargs)
    return control


def connected_console(console_class=RichIPythonWidget, **kwargs):
    """Create a console widget, connected to another kernel running in
       the current process

    This only works on IPython v1.0 and above

    Parameters
    ----------
    console_class : The class of the console widget to create
    kwargs : Extra variables to put into the namespace
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


glue_banner_parts = []
glue_banner_parts.append("Glue %s " % __version__)
glue_banner_parts.append("Predefined variables - drag additional items into "
                         "this window to use:")
glue_banner_parts.append("\t* data_collection (aliased to dc)")
glue_banner_parts.append("\t* application")
glue_banner_parts.append("\t* hub")

glue_banner = '\n'.join(glue_banner_parts)


class DragAndDropTerminal(RichIPythonWidget):
    banner = default_banner + '\n' + glue_banner

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


# Works for IPython 0.12, 0.13
def default_kernel_app():
    """ Return a configured IPKernelApp """

    def event_loop(kernel):
        """ Non-blocking qt event loop."""
        kernel.timer = QtCore.QTimer()
        kernel.timer.timeout.connect(kernel.do_one_iteration)
        kernel.timer.start(1000 * kernel._poll_interval)

    app = IPKernelApp.instance()
    try:
        app.initialize(['python', '--pylab=qt'])
    except ZMQError:
        pass  # already set up

    app.kernel.eventloop = event_loop

    try:
        app.start()
    except RuntimeError:  # already started
        pass

    return app


def default_manager(kernel):
    """ Return a configured QtKernelManager

    :param kernel: An IPKernelApp instance
    """
    connection_file = find_connection_file(kernel.connection_file)
    manager = QtKernelManager(connection_file=connection_file)
    manager.load_connection_file()
    manager.start_channels()
    atexit.register(manager.cleanup_connection_file)
    return manager


def _glue_terminal_1(**kwargs):
    """ Used for IPython v0.13, v0.12
    """
    kernel_app = default_kernel_app()
    manager = default_manager(kernel_app)

    try:  # IPython v0.13
        widget = DragAndDropTerminal(gui_completion='droplist')
    except TraitError:  # IPython v0.12
        widget = DragAndDropTerminal(gui_completion=True)
    widget.kernel_manager = manager
    widget.shell = kernel_app.shell

    # update namespace
    widget.update_namespace(kwargs)

    # IPython v0.12 turns on MPL interactive. Turn it back off
    import matplotlib
    matplotlib.interactive(False)
    return widget


# works on IPython v0.13, v0.14
@contextmanager
def redirect_output(session, pub_socket):
    """Prevent any of the widgets from permanently hijacking stdout or
    stderr"""
    sys.stdout = OutStream(session, pub_socket, u'stdout')
    sys.stderr = OutStream(session, pub_socket, u'stderr')
    try:
        yield
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


def non_blocking_eventloop(kernel):
    kernel.timer = QtCore.QTimer()
    kernel.timer.timeout.connect(kernel.do_one_iteration)
    kernel.timer.start(1000 * kernel._poll_interval)


class EmbeddedQtKernel(Kernel):

    def __init__(self, *args, **kwargs):
        super(EmbeddedQtKernel, self).__init__(*args, **kwargs)
        self.eventloop = non_blocking_eventloop

    def do_one_iteration(self):
        with redirect_output(self.session, self.iopub_socket):
            super(EmbeddedQtKernel, self).do_one_iteration()

    def execute_request(self, stream, ident, parent):
        with redirect_output(self.session, self.iopub_socket):
            super(EmbeddedQtKernel, self).execute_request(
                stream, ident, parent)


class EmbeddedQtKernelApp(IPKernelApp):

    def init_kernel(self):
        shell_stream = ZMQStream(self.shell_socket)
        kernel = EmbeddedQtKernel(config=self.config, session=self.session,
                                  shell_streams=[shell_stream],
                                  iopub_socket=self.iopub_socket,
                                  stdin_socket=self.stdin_socket,
                                  log=self.log,
                                  profile_dir=self.profile_dir,
                                  )
        self.kernel = kernel
        kernel.record_ports(self.ports)

    def start(self):
        # handoff between IOLoop and QApplication event loops
        loop = ioloop.IOLoop.instance()
        # We used to have a value of 0ms as the second argument
        # (callback_time) in the following call, but this caused the
        # application to hang on certain setups, so use 1ms instead.
        stopper = ioloop.PeriodicCallback(loop.stop, 1, loop)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(loop.start)
        self.timer.start(100)
        stopper.start()
        super(EmbeddedQtKernelApp, self).start()


class EmbeddedIPythonWidget(DragAndDropTerminal):
    gui_completion = 'droplist'

    def __init__(self, **kwargs):
        super(EmbeddedIPythonWidget, self).__init__(**kwargs)
        self._init_kernel_app()
        self._init_kernel_manager()
        self.update_namespace(kwargs)

    def _init_kernel_app(self):
        app = EmbeddedQtKernelApp.instance()
        try:
            app.initialize([])
        except ZMQError:
            pass  # already set up
        try:
            app.start()
        except RuntimeError:  # already started
            pass
        self.app = app
        self.shell = app.shell

    def _init_kernel_manager(self):
        connection_file = find_connection_file(self.app.connection_file)
        manager = QtKernelManager(connection_file=connection_file)
        manager.load_connection_file()
        manager.start_channels()
        atexit.register(manager.cleanup_connection_file)
        self.kernel_manager = manager

    def update_namespace(self, ns):
        self.app.shell.user_ns.update(ns)


def _glue_terminal_2(**kwargs):
    """Used for IPython v0.13, v0.14"""
    return EmbeddedIPythonWidget(**kwargs)


def _glue_terminal_3(**kwargs):
    """Used for IPython v1.0 and beyond

    :param kwargs: Keywords which are passed to Widget init,
    and which are also passed to the current namespace
    """
    # see IPython/docs/examples/frontends/inprocess_qtconsole.p

    shell = get_ipython()
    if shell is None or isinstance(shell, InProcessInteractiveShell):
        return in_process_console(console_class=DragAndDropTerminal, **kwargs)
    return connected_console(console_class=DragAndDropTerminal, **kwargs)


def glue_terminal(**kwargs):
    """ Return a qt widget which embed an IPython interpreter.

        Extra keywords will be added to the namespace of the shell

        :param kwargs: Extra variables to be added to the namespace

        :rtype: QWidget
    """
    from distutils.version import LooseVersion
    import IPython
    ver = LooseVersion(IPython.__version__)
    v1_0 = LooseVersion('1.0')
    v0_12 = LooseVersion('0.12')
    v0_13 = LooseVersion('0.13')

    if ver >= v1_0:
        return _glue_terminal_3(**kwargs)
    if ver >= v0_13:
        return _glue_terminal_2(**kwargs)
    if ver >= v0_12:
        return _glue_terminal_1(**kwargs)

    raise RuntimeError("Glue terminal requires IPython >= 0.12")
