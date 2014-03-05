import numpy as np

from ..external.qt.QtCore import Qt
from ..clients.profile_viewer import ProfileViewer, SliderArtist
from .widgets.mpl_widget import MplWidget
from .mouse_mode import SpectrumExtractorMode
from ..core.callback_property import add_callback

class SpectrumTool(object):

    def __init__(self, image_widget):
        self.image_widget = image_widget
        self.widget = MplWidget()
        self.widget.setWindowFlags(Qt.Tool)
        self.client = self.image_widget.client
        self.axes = self.widget.canvas.fig.add_subplot(111)
        self.profile = ProfileViewer(self.axes)
        self.mouse_mode = self._setup_mouse_mode()
        self._setup_handles()

    def _setup_mouse_mode(self):
        mode = SpectrumExtractorMode(self.image_widget.client.axes,
                                     release_callback=self._update_profile)
        return mode

    def _setup_handles(self):
        self.slice_handle = self.profile.new_slider_handle(self._set_slice)
        self.slicer_artist = SliderArtist(self.slice_handle)
        set_slicer = lambda *args: setattr(self.slice_handle,
                                           'value', self.current_channel)
        add_callback(self.client, 'slice', set_slicer)

    @property
    def current_channel(self):
        s = self.client.slice
        assert len(s) == 3
        for ss in s:
            if ss not in ['x', 'y']:
                return ss

    def _recenter_handles(self):
        self.slice_handle.value = sum(self.axes.get_xlim()) / 2.

    def _set_slice(self, value):
        slc = list(self.client.slice)
        assert len(slc) == 3

        for i, s in enumerate(slc):
            if s not in ['x', 'y']:
                slc[i] = int(value)

        self.client.slice = tuple(slc)

    def _update_profile(self, *args):
        data = self.client.display_data
        att = self.client.display_attribute
        if data is None or att is None:
            return

        roi = self.mouse_mode.roi()
        x, y = self._extract_spectrum(roi, data, att)
        xlim = self.axes.get_xlim()
        self.profile.set_profile(x, y, c='k')
        if self.axes.get_xlim() != xlim:
            self._recenter_handles()
        self.show()

    def _extract_spectrum(self, roi, data, att):
        #XXX this should be an roi/data method
        #XXX assuming roi is rectangular
        assert data.ndim == 3

        l, r, b, t = roi.xmin, roi.xmax, roi.ymin, roi.ymax
        slc = [slice(None) for _ in range(data.ndim)]
        for i, s in enumerate(self.client.slice):
            if s == 'x':
                slc[i] = slice(l, r)
            elif s == 'y':
                slc[i] = slice(b, t)

        data = data[tuple([att] + slc)]
        for i, s in reversed(list(enumerate(self.client.slice))):
            if s in ['x', 'y']:
                data = np.nansum(data, axis=i) / np.isfinite(data).sum(axis=i)

        data = data.ravel()
        return np.arange(data.size), data

    def show(self):
        self.widget.show()
