from glue.backends import TimerBase

from glue.external.qt import QtCore


class Timer(TimerBase):

    def __init__(self, interval, callback):
        self._timer = QtCore.QTimer()
        self._timer.setInterval(interval)
        self._timer.timeout.connect(callback)

    def start(self):
        self._timer.start()

    def stop(self):
        self._timer.stop()
