from ..backends import TimerBase

from ..external.qt.QtCore import QTimer


class Timer(TimerBase):

    def __init__(self, interval, callback):
        self._timer = QTimer()
        self._timer.setInterval(interval)
        self._timer.timeout.connect(callback)

    def start(self):
        self._timer.start()

    def stop(self):
        self._timer.stop()
