"""
A common interface for accessing backend UI functionality.

At the moment, the only backend is Qt
"""
from abc import abstractmethod
_backend = None


class TimerBase(object):

    @abstractmethod
    def __init__(self, interval, callback):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def start(self):
        pass


class QtTimer(TimerBase):

    def __init__(self, interval, callback):
        from qtpy import QtCore
        self._timer = QtCore.QTimer()
        self._timer.setInterval(interval)
        self._timer.timeout.connect(callback)

    def start(self):
        self._timer.start()

    def stop(self):
        self._timer.stop()


def get_timer(backend='qt'):

    if backend == 'qt':
        return QtTimer
    else:
        raise ValueError("Only QT Backend supported")


def get_backend(backend='qt'):
    global _backend

    if _backend is not None:
        return _backend

    if backend != 'qt':
        raise ValueError("Only QT Backend supported")

    from glue.qt import qt_backend

    _backend = qt_backend
    return _backend
