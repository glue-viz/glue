import numpy as np

from ..external.qt.QtCore import Qt
from ..external.qt.QtGui import QMainWindow
from ..clients.profile_viewer import ProfileViewer, SliderArtist
from .widgets.mpl_widget import MplWidget
from .mouse_mode import SpectrumExtractorMode
from ..core.callback_property import add_callback
from .glue_toolbar import GlueToolbar


class Extractor(object):

    @staticmethod
    def abcissa(data, axis):
        slc = [0 for _ in data.shape]
        slc[axis] = slice(None, None)
        att = data.get_world_component_id(axis)
        return data[att, tuple(slc)].ravel()

    @staticmethod
    def spectrum(data, attribute, roi, xaxis, yaxis, zaxis):
        l, r, b, t = roi.xmin, roi.xmax, roi.ymin, roi.ymax
        slc = [slice(None) for _ in data.shape]
        slc[xaxis] = slice(l, r)
        slc[yaxis] = slice(b, t)

        x = Extractor.abcissa(data, zaxis)

        data = data[attribute, tuple(slc)]
        for i in reversed(list(range(data.ndim))):
            if i != zaxis:
                data = np.nansum(data, axis=i) / np.isfinite(data).sum(axis=i)

        data = data.ravel()
        return x, data

    @staticmethod
    def world2pixel(data, axis, value):
        x = Extractor.abcissa(data, axis)
        if x.size > 1 and (x[1] < x[0]):
            x = x[::-1]
            result = x.size - np.searchsorted(x, value) - 1
        else:
            result = np.searchsorted(x, value)
        return np.clip(result, 0, x.size - 1)

    @staticmethod
    def pixel2world(data, axis, value):
        x = Extractor.abcissa(data, axis)
        return x[np.clip(value, 0, x.size - 1)]


class SpectrumTool(object):

    def __init__(self, image_widget):
        self.image_widget = image_widget
        self.widget = QMainWindow()
        self.widget.setWindowFlags(Qt.Tool)
        w = MplWidget()

        self.widget.setCentralWidget(w)
        self.canvas = w.canvas

        self.client = self.image_widget.client
        self.axes = self.canvas.fig.add_subplot(111)
        self.profile = ProfileViewer(self.axes)
        self.mouse_mode = self._setup_mouse_mode()

        self._last_profile_axis = None
        self._relim_requested = False

        self._setup_handles()
        self._setup_toolbar()
        self._setup_double_click_handler()

    def _setup_mouse_mode(self):
        mode = SpectrumExtractorMode(self.image_widget.client.axes,
                                     release_callback=self._update_profile)

        def toggle_mode_enabled(data):
            mode.enabled = (data.ndim > 2)

        add_callback(self.client, 'display_data', toggle_mode_enabled)
        return mode

    def _setup_double_click_handler(self):
        def _check_recenter(event):
            if event.dblclick:
                self.slice_handle.value = event.xdata

        self.canvas.mpl_connect('button_press_event',
                                _check_recenter)

    def _setup_toolbar(self):
        tb = GlueToolbar(self.canvas, self.widget)

        # disable ProfileViewer mouse processing during mouse modes
        tb.mode_activated.connect(self.profile.disconnect)
        tb.mode_deactivated.connect(self.profile.connect)

        self.widget.addToolBar(tb)
        return tb

    def _setup_handles(self):

        def _set_client_from_handle(value):
            """Update client.slice given handle value"""
            slc = list(self.client.slice)

            # client.slice stored in pixel coords
            value = Extractor.world2pixel(
                self.data,
                self.profile_axis, value)
            slc[self.profile_axis] = value
            self.client.slice = tuple(slc)

        def _set_handle_from_client(slc):
            """Update handle.value given client.slice"""
            # handle.value is stored in world coordinates
            val = slc[self.profile_axis]
            val = Extractor.pixel2world(self.data, self.profile_axis, val)
            self.slice_handle.value = val

        self.slice_handle = self.profile.new_slider_handle()
        self.slicer_artist = SliderArtist(self.slice_handle)

        add_callback(self.client, 'slice', _set_handle_from_client)
        add_callback(self.slice_handle, 'value', _set_client_from_handle)
        add_callback(self.client, 'slice', self._check_invalidate)

    def _check_invalidate(self, slc):
        if self.profile_axis is None:
            return

        if self._last_profile_axis != self.profile_axis:
            self._last_profile_axis = self.profile_axis
            self.reset()

    def reset(self):
        self.hide()
        self.mouse_mode.clear()
        self._relim_requested = True

    @property
    def data(self):
        return self.client.display_data

    @property
    def profile_axis(self):
        # XXX make this settable
        # defaults to the non-xy axis with the most channels
        slc = self.client.slice
        candidates = [i for i, s in enumerate(slc) if s not in ['x', 'y']]
        return max(candidates, key=lambda i: self.data.shape[i])

    def _recenter_handles(self):
        self.slice_handle.value = sum(self.axes.get_xlim()) / 2.

    def _update_profile(self, *args):
        data = self.data
        att = self.client.display_attribute
        slc = self.client.slice
        roi = self.mouse_mode.roi()

        xid = data.get_world_component_id(self.profile_axis)
        units = data.get_component(xid).units
        xlabel = str(xid) if units is None else '%s [%s]' % (xid, units)

        if data is None or att is None:
            return

        xax = slc.index('x')
        yax = slc.index('y')
        zax = self.profile_axis

        x, y = Extractor.spectrum(data, att, roi, xax, yax, zax)

        xlim = self.axes.get_xlim()
        self.profile.set_profile(x, y, c='k')

        # relim x range if requested
        if self._relim_requested:
            self._relim_requested = False
            self.axes.set_xlim(np.nanmin(x), np.nanmax(x))

        # relim y range to data within the view window
        mask = (self.axes.get_xlim()[0] <= x) & (x <= self.axes.get_xlim()[1])
        ymask = y[mask]
        ylim = np.nanmin(ymask), np.nanmax(ymask)
        self.axes.set_ylim(ylim[0], ylim[1] + .05 * (ylim[1] - ylim[0]))

        if self.axes.get_xlim() != xlim:
            self._recenter_handles()

        self.axes.set_xlabel(xlabel)
        self.axes.figure.canvas.draw()
        self.show()

    def _move_below_image_widget(self):
        rect = self.image_widget.frameGeometry()
        pos = self.image_widget.mapToGlobal(rect.bottomLeft())
        self.widget.setGeometry(pos.x(), pos.y(),
                                rect.width(), rect.width() / 2.5)

    def show(self):
        if self.widget.isVisible():
            return
        self._move_below_image_widget()
        self.widget.show()

    def hide(self):
        self.widget.close()
