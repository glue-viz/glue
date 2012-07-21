"""
A GUI Ipython terminal window which can interact
with Glue. Based on code from

http://stackoverflow.com/a/9796491/1332492
and
http://stackoverflow.com/a/11525205/1332492

Usage:
   new_widget = glue_terminal(**kwargs)
"""
import atexit

from PyQt4 import QtCore

from zmq import ZMQError
from IPython.zmq.ipkernel import IPKernelApp
from IPython.lib.kernel import find_connection_file
from IPython.frontend.qt.kernelmanager import QtKernelManager
from IPython.frontend.qt.console.rich_ipython_widget import RichIPythonWidget
from IPython.utils.traitlets import TraitError

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


def glue_terminal(**kwargs):
    """ Return a qt widget which embed an IPython interpreter.

    Extra keywords will be added to the namespace of the shell

    :param kwargs: Extra variables to be added to the namespace

    :rtype: QWidget
    """
    kernel_app = default_kernel_app()
    manager = default_manager(kernel_app)

    try:  # IPython v0.13
        widget = RichIPythonWidget(gui_completion='droplist')
    except TraitError:  # IPython v0.12
        widget = RichIPythonWidget(gui_completion=True)
    widget.kernel_manager = manager

    # update namespace
    kernel_app.shell.user_ns.update(kwargs)

    # for debugging
    kernel_app.shell.user_ns['_kernel'] = kernel_app
    kernel_app.shell.user_ns['_manager'] = manager
    kernel_app.shell.user_ns['_widget'] = glue_terminal

    return widget
