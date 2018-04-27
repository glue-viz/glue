from __future__ import absolute_import, division, print_function

from qtpy import QtCore

__all__ = ['Worker']


class Worker(QtCore.QThread):

    result = QtCore.Signal(object)
    error = QtCore.Signal(object)

    def __init__(self, func, *args, **kwargs):
        """
        Execute a function call on a different QThread

        :param func: The function object to call
        :param args: arguments to pass to the function
        :param kwargs: kwargs to pass to the function
        """
        super(Worker, self).__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        """
        Invoke the function
        Upon successful completion, the result signal will be fired
        with the output of the function
        If an exception occurs, the error signal will be fired with
        the result form sys.exc_infno()
        """
        try:
            result = self.func(*self.args, **self.kwargs)
            self.result.emit(result)
        except Exception:
            import sys
            self.error.emit(sys.exc_info())
