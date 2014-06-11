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


def get_backend(backend='qt'):
    global _backend

    if _backend is not None:
        return _backend

    if backend != 'qt':
        raise ValueError("Only QT Backend supported")

    from .qt import qt_backend

    _backend = qt_backend
    return _backend
