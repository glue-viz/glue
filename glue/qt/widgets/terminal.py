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

import sys
import atexit

from ...external.qt import QtCore
from ...external.qt.QtGui import QInputDialog

from zmq import ZMQError
from zmq.eventloop.zmqstream import ZMQStream

try:  # IPython <= 0.13.2
    from IPython.zmq.ipkernel import IPKernelApp, Kernel
    from IPython.zmq.iostream import OutStream
    from IPython.frontend.qt.kernelmanager import QtKernelManager
except ImportError:  # IPython >= 1.0dev
    from IPython.kernel.zmq.ipkernel import Kernel
    from IPython.kernel.zmq.kernelapp import IPKernelApp
    from IPython.kernel.zmq.iostream import OutStream
    from IPython.frontend.qt.manager import QtKernelManager

from IPython.lib.kernel import find_connection_file
from IPython.frontend.qt.console.rich_ipython_widget import RichIPythonWidget

from contextlib import contextmanager
from zmq.eventloop import ioloop
from IPython.utils.traitlets import TraitError


class DragAndDropTerminal(RichIPythonWidget):
    def __init__(self, **kwargs):
        super(DragAndDropTerminal, self).__init__(**kwargs)
        self.setAcceptDrops(True)
        self.shell = None

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

        var, ok = QInputDialog.getText(self, "Choose a variable name",
                                       "Choose a variable name", text="x")
        if ok:
            #unpack single-item lists for convenience
            if isinstance(obj, list) and len(obj) == 1:
                obj = obj[0]

            var = {str(var): obj}
            self.update_namespace(var)
            event.accept()
        else:
            event.ignore()


#Works for IPython 0.12, 0.13
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

    #IPython v0.12 turns on MPL interactive. Turn it back off
    import matplotlib
    matplotlib.interactive(False)
    return widget


#works on IPython v0.13, v0.14
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
        #handoff between IOLoop and QApplication event loops
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
    # see IPython/docs/examples/frontends/inprocess_qtconsole.py

    from IPython.kernel.inprocess.ipkernel import InProcessKernel
    from IPython.frontend.qt.inprocess import \
        QtInProcessKernelManager

    km = QtInProcessKernelManager()
    km.start_kernel()

    kernel = km.kernel
    kernel.gui = 'qt4'

    client = km.client()
    client.start_channels()

    control = DragAndDropTerminal()
    control.kernel_manager = km
    control.kernel_client = client
    control.shell = kernel.shell
    control.update_namespace(kwargs)

    return control


def glue_terminal(**kwargs):
    """ Return a qt widget which embed an IPython interpreter.

        Extra keywords will be added to the namespace of the shell

        :param kwargs: Extra variables to be added to the namespace

        :rtype: QWidget
    """
    import IPython
    rels = IPython.__version__.split('.')
    rel = int(rels[0])
    maj = int(rels[1])
    if rel == 0 and maj < 12:
        raise RuntimeError("Glue terminal requires IPython >= 0.12")

    if rel >= 1:
        return _glue_terminal_3(**kwargs)
    if rel == 0 and maj >= 13:
        return _glue_terminal_2(**kwargs)
    else:
        return _glue_terminal_1(**kwargs)
