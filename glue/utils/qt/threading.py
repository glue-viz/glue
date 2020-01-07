import sys

from qtpy import QtCore

__all__ = ['Worker']


class Worker(QtCore.QThread):

    result = QtCore.Signal(object)
    error = QtCore.Signal(object)

    def __init__(self, func, *args, **kwargs):
        """
        Execute a function call on a different thread.

        Parameters
        ----------
        func : callable
            The function object to call
        args
            Positional arguments to pass to the function
        kwargs
            Keyword arguments to pass to the function
        """
        super(Worker, self).__init__()
        self.func = func
        self.args = args or ()
        self.kwargs = kwargs or {}

    def run(self):
        """
        Invoke the function. Upon successful completion, the result signal
        will be fired with the output of the function If an exception
        occurs, the error signal will be fired with the result form
        ``sys.exc_info()``.
        """
        try:
            self.running = True
            result = self.func(*self.args, **self.kwargs)
            self.running = False
            self.result.emit(result)
        except Exception:
            self.running = False
            self.error.emit(sys.exc_info())
