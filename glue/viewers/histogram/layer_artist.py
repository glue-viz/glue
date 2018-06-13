from __future__ import absolute_import, division, print_function

import sys
import time
import queue
import warnings

import numpy as np
from matplotlib.patches import Rectangle

from glue.utils import defer_draw, queue_to_list

from glue.viewers.histogram.state import HistogramLayerState
from glue.viewers.histogram.python_export import python_export_histogram_layer
from glue.viewers.matplotlib.layer_artist import MatplotlibLayerArtist
from glue.core.exceptions import IncompatibleAttribute

try:
    import qtpy  # noqa
except Exception:
    QT_INSTALLED = False
else:
    QT_INSTALLED = True

if QT_INSTALLED:

    # When using Qt, we make use of a thread that continuously listens for
    # requests to update the histogram and we run these as needed. In future,
    # we should add the ability to interrupt compute jobs if a newer compute
    # job is requested.

    from qtpy.QtCore import Signal, QThread

    class ComputeWorker(QThread):

        compute_start = Signal()
        compute_end = Signal()
        compute_error = Signal(object)

        def __init__(self, function):
            super(ComputeWorker, self).__init__()
            self.function = function
            self.running = False

        def run(self):
            self.function()


class HistogramLayerArtist(MatplotlibLayerArtist):

    _layer_state_cls = HistogramLayerState
    _python_exporter = python_export_histogram_layer

    def __init__(self, axes, viewer_state, layer_state=None, layer=None):

        super(HistogramLayerArtist, self).__init__(axes, viewer_state,
                                                   layer_state=layer_state, layer=layer)

        # Watch for changes in the viewer state which would require the
        # layers to be redrawn
        self._viewer_state.add_global_callback(self._update_histogram)
        self.state.add_global_callback(self._update_histogram)

        self.reset_cache()

        if QT_INSTALLED:
            self.setup_thread()

    def remove(self):
        super(HistogramLayerArtist, self).remove()
        if QT_INSTALLED and self._worker is not None:
            self._work_queue.put('stop')
            self._worker.exit()
            # Need to wait otherwise the thread will be destroyed while still
            # running, causing a segmentation fault
            self._worker.wait()
            self._worker = None

    @property
    def is_computing(self):
        if QT_INSTALLED:
            return self._worker.running

    def reset_cache(self):
        self._last_viewer_state = {}
        self._last_layer_state = {}

    def setup_thread(self):
        self._worker = ComputeWorker(self._thread_loop)
        self._worker.compute_end.connect(self._calculate_histogram_postthread)
        self._worker.compute_error.connect(self._calculate_histogram_error)
        self._worker.compute_start.connect(self.notify_start_computation)
        self._work_queue = queue.Queue()
        self._worker.start()

    def _thread_loop(self):

        error = None

        while True:

            time.sleep(1 / 25)

            msgs = queue_to_list(self._work_queue)

            if 'stop' in msgs:
                return
            elif len(msgs) == 0:
                # We change this here rather than in the try...except below
                # to avoid stopping and starting in quick succession.
                if self._worker.running:
                    self._worker.running = False
                    if error is None:
                        self._worker.compute_end.emit()
                    else:
                        self._worker.compute_error.emit(error)
                        error = None
                continue

            # If any resets were requested, honor this
            reset = any(msgs)

            try:
                self._worker.running = True
                self._worker.compute_start.emit()
                self._calculate_histogram_thread(reset=reset)
            except Exception:
                error = sys.exc_info()
            else:
                error = None

    @defer_draw
    def _calculate_histogram(self, reset=False):
        if QT_INSTALLED:
            self._work_queue.put(reset)
        else:
            try:
                self.notify_start_computation()
                self._calculate_histogram_thread(reset=reset)
            except Exception:
                self._calculate_histogram_error(sys.exc_info())
            else:
                self._calculate_histogram_postthread()

    def _calculate_histogram_thread(self, reset=False):
        # We need to ignore any warnings that happen inside the thread
        # otherwise the thread tries to send these to the glue logger (which
        # uses Qt), which then results in this kind of error:
        # QObject::connect: Cannot queue arguments of type 'QTextCursor'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if reset:
                self.state.reset_cache()
            self.state.update_histogram()

    def _calculate_histogram_postthread(self):
        self.notify_end_computation()
        self._update_artists()
        self._update_visual_attributes()

    @defer_draw
    def _calculate_histogram_error(self, exc):
        self.notify_end_computation()
        self.redraw()
        if issubclass(exc[0], (IncompatibleAttribute, IndexError)):
            self.disable_invalid_attributes(self._viewer_state.x_att)

    @defer_draw
    def _update_artists(self):

        mpl_hist_edges, mpl_hist = self.state.histogram

        if mpl_hist_edges.size == 0 or mpl_hist.sum() == 0:
            return

        if len(self.mpl_artists) > len(mpl_hist):
            for artist in self.mpl_artists[len(mpl_hist):]:
                artist.remove()
            self.mpl_artists = self.mpl_artists[:len(mpl_hist)]
        elif len(self.mpl_artists) < len(mpl_hist):
            for i in range(len(mpl_hist) - len(self.mpl_artists)):
                artist = Rectangle((0, 0), 0, 0)
                self.mpl_artists.append(artist)
                self.axes.add_artist(artist)
            self._update_visual_attributes()

        widths = np.diff(mpl_hist_edges)

        bottom = 0 if not self._viewer_state.y_log else 1e-100

        for mpl_artist, x, y, dx in zip(self.mpl_artists, mpl_hist_edges[:-1], mpl_hist, widths):
            mpl_artist.set_width(dx)
            mpl_artist.set_height(y)
            mpl_artist.set_xy((x, bottom))

        # TODO: move the following to state

        # We have to do the following to make sure that we reset the y_max as
        # needed. We can't simply reset based on the maximum for this layer
        # because other layers might have other values, and we also can't do:
        #
        #   self._viewer_state.y_max = max(self._viewer_state.y_max, result[0].max())
        #
        # because this would never allow y_max to get smaller.

        self.state._y_max = mpl_hist.max()
        if self._viewer_state.y_log:
            self.state._y_max *= 2
        else:
            self.state._y_max *= 1.2

        if self._viewer_state.y_log:
            self.state._y_min = mpl_hist[mpl_hist > 0].min() / 10
        else:
            self.state._y_min = 0

        largest_y_max = max(getattr(layer, '_y_max', 0) for layer in self._viewer_state.layers)
        if largest_y_max != self._viewer_state.y_max:
            self._viewer_state.y_max = largest_y_max

        smallest_y_min = min(getattr(layer, '_y_min', np.inf) for layer in self._viewer_state.layers)
        if smallest_y_min != self._viewer_state.y_min:
            self._viewer_state.y_min = smallest_y_min

        self.redraw()

    @defer_draw
    def _update_visual_attributes(self):

        if not self.enabled:
            return

        for mpl_artist in self.mpl_artists:
            mpl_artist.set_visible(self.state.visible)
            mpl_artist.set_zorder(self.state.zorder)
            mpl_artist.set_edgecolor('none')
            mpl_artist.set_facecolor(self.state.color)
            mpl_artist.set_alpha(self.state.alpha)

        self.redraw()

    def _update_histogram(self, force=False, **kwargs):

        if (self._viewer_state.hist_x_min is None or
                self._viewer_state.hist_x_max is None or
                self._viewer_state.hist_n_bin is None or
                self._viewer_state.x_att is None or
                self.state.layer is None):
            return

        # Figure out which attributes are different from before. Ideally we shouldn't
        # need this but currently this method is called multiple times if an
        # attribute is changed due to x_att changing then hist_x_min, hist_x_max, etc.
        # If we can solve this so that _update_histogram is really only called once
        # then we could consider simplifying this. Until then, we manually keep track
        # of which properties have changed.

        changed = set()

        if not force:

            for key, value in self._viewer_state.as_dict().items():
                if value != self._last_viewer_state.get(key, None):
                    changed.add(key)

            for key, value in self.state.as_dict().items():
                if value != self._last_layer_state.get(key, None):
                    changed.add(key)

        self._last_viewer_state.update(self._viewer_state.as_dict())
        self._last_layer_state.update(self.state.as_dict())

        if force or any(prop in changed for prop in ('layer', 'x_att', 'hist_x_min', 'hist_x_max', 'hist_n_bin', 'x_log', 'y_log', 'normalize', 'cumulative')):
            self._calculate_histogram(reset=force)

        if force or any(prop in changed for prop in ('alpha', 'color', 'zorder', 'visible')):
            self._update_visual_attributes()

    @defer_draw
    def update(self):
        self.state.reset_cache()
        self._update_histogram(force=True)
        self.redraw()
