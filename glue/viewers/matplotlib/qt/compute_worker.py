import sys
import time
import queue

from glue.utils import queue_to_list
from qtpy.QtCore import Signal, QThread


# For some viewers, we make use of a thread that continuously listens for
# requests to update the profile and we run these as needed. In future,
# we should add the ability to interrupt compute jobs if a newer compute
# job is requested.


class ComputeWorker(QThread):

    compute_start = Signal()
    compute_end = Signal()
    compute_error = Signal(object)

    def __init__(self, function):
        super(ComputeWorker, self).__init__()
        self.function = function
        self.running = False
        self.work_queue = queue.Queue()

    def run(self):

        error = None

        while True:

            time.sleep(1 / 25)

            msgs = queue_to_list(self.work_queue)

            if 'stop' in msgs:
                return

            elif len(msgs) == 0:
                # We change this here rather than in the try...except below
                # to avoid stopping and starting in quick succession.
                if self.running:
                    self.running = False
                    if error is None:
                        self.compute_end.emit()
                    else:
                        self.compute_error.emit(error)
                        error = None
                continue

            # If any resets were requested, honor this
            reset = any(msgs)

            try:
                self.running = True
                self.compute_start.emit()
                self.function(reset=reset)
            except Exception:
                error = sys.exc_info()
            else:
                error = None
