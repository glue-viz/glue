import numpy as np

from ..external.qt.QtCore import Qt
from ..external.qt.QtGui import (QMainWindow, QWidget,
                                 QHBoxLayout, QTabWidget,
                                 QComboBox, QFormLayout, QPushButton)

from ..clients.profile_viewer import ProfileViewer
from .widgets.mpl_widget import MplWidget
from .mouse_mode import SpectrumExtractorMode
from ..core.callback_property import add_callback, ignore_callback
from ..core.util import Pointer
from .glue_toolbar import GlueToolbar
from .qtutil import load_ui, nonpartial
from .widget_properties import CurrentComboProperty
from ..core.fitters import AstropyModelFitter
from ..core.aggregate import Aggregate


class Extractor(object):
    # Warning:
    # Coordinate conversion is not well-defined if pix2world is not
    # monotonic!

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
        shp = data.shape
        l, r = np.clip([l, r], 0, shp[xaxis])
        b, t = np.clip([b, t], 0, shp[yaxis])
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


class SpectrumContext(object):
    client = Pointer('main.client')
    data = Pointer('main.data')
    profile_axis = Pointer('main.profile_axis')
    canvas = Pointer('main.canvas')
    profile = Pointer('main.profile')

    def __init__(self, main):

        self.main = main
        self.grip = None
        self.panel = None
        self.widget = None

        self._setup_grip()
        self._setup_widget()
        self._connect()

    def _setup_grip(self):
        raise NotImplementedError()

    def _setup_widget(self):
        raise NotImplementedError()

    def _connect(self):
        pass

    def set_enabled(self, enabled):
        self.enable() if enabled else self.disable()

    def enable(self):
        if self.grip is not None:
            self.grip.enable()

    def disable(self):
        if self.grip is not None:
            self.grip.disable()

    def recenter(self, lim):
        """Re-center the grip to the given x axlis limit tuple"""
        if self.grip is None:
            return
        if hasattr(self.grip, 'value'):
            self.grip.value = sum(lim) / 2.
            return

        # Range grip
        cen = sum(lim) / 2
        wid = max(lim) - min(lim)
        self.grip.range = cen - wid / 4, cen + wid / 4


class NavContext(SpectrumContext):

    def _setup_grip(self):
        def _set_client_from_grip(value):
            """Update client.slice given grip value"""
            slc = list(self.client.slice)

            # client.slice stored in pixel coords
            value = Extractor.world2pixel(
                self.data,
                self.profile_axis, value)
            slc[self.profile_axis] = value

            self.client.slice = tuple(slc)

        def _set_grip_from_client(slc):
            """Update grip.value given client.slice"""
            # grip.value is stored in world coordinates
            val = slc[self.profile_axis]
            val = Extractor.pixel2world(self.data, self.profile_axis, val)

            # If pix2world not monotonic, this can trigger infinite recursion.
            # Avoid by disabling callback loop
            # XXX better to specifically ignore _set_client_from_grip
            with ignore_callback(self.client, 'slice'):
                self.grip.value = val

        self.grip = self.main.profile.new_value_grip()

        add_callback(self.client, 'slice', _set_grip_from_client)
        add_callback(self.grip, 'value', _set_client_from_grip)

    def _connect(self):
        pass

    def _setup_widget(self):
        self.widget = QWidget()


class CollapseContext(SpectrumContext):

    def _setup_grip(self):
        self.grip = self.main.profile.new_range_grip()

    def _setup_widget(self):
        w = QWidget()
        l = QFormLayout()
        w.setLayout(l)

        combo = QComboBox()
        combo.addItem("Mean", userData=Aggregate.mean)
        combo.addItem("Median", userData=Aggregate.median)
        combo.addItem("Max", userData=Aggregate.max)
        combo.addItem("Centroid", userData=Aggregate.mom1)
        combo.addItem("Linewidth", userData=Aggregate.mom2)
        combo.addItem("Coordinate of maximum", userData=Aggregate.argmax)
        combo.addItem("Coordinate of minimum", userData=Aggregate.argmin)

        run = QPushButton("Collapse")
        self._run = run

        l.addRow("Collapse Function", combo)
        l.addRow("", run)
        self.widget = w
        self._combo = combo

    def _connect(self):
        self._run.clicked.connect(nonpartial(self._aggregate))

    def _aggregate(self):
        func = self._combo.itemData(self._combo.currentIndex())

        rng = Extractor.world2pixel(self.data,
                                    self.profile_axis,
                                    self.grip.range)

        agg = Aggregate(self.data, self.client.display_attribute,
                        self.main.profile_axis, self.client.slice, rng)

        im = func(agg)
        self.client.override_image(im)


class FitContext(SpectrumContext):
    error = CurrentComboProperty('ui.uncertainty_combo')
    fitter = CurrentComboProperty('ui.profile_combo')

    def _setup_grip(self):
        self.grip = self.main.profile.new_range_grip()

    def _setup_widget(self):
        self.ui = load_ui('spectrum_fit_panel')
        self.widget = self.ui

    @property
    def fitter(self):
        x, y = self.main.profile.profile_data(xlim=self.grip.range)
        amp = y.max()
        y = y / y.sum()

        mean = (x * y).sum()
        stddev = np.sqrt((y * (x - mean) ** 2).sum())
        return AstropyModelFitter.gaussian_fitter(amplitude=amp,
                                                  mean=mean,
                                                  stddev=stddev)

    def _connect(self):
        self.ui.fit_button.clicked.connect(nonpartial(self.fit))

    def fit(self):
        xlim = self.grip.range
        fitter = self.fitter
        fit = self.main.profile.fit(fitter, xlim=xlim)
        self._report_fit(fit)
        self.canvas.draw()

    def _report_fit(self, fit):
        self.ui.results_box.document().setPlainText(str(fit))


class SpectrumTool(object):

    def __init__(self, image_widget):
        self._relim_requested = True

        self.image_widget = image_widget
        self._build_main_widget()

        self.client = self.image_widget.client
        self.profile = ProfileViewer(self.canvas.fig)
        self.axes = self.profile.axes

        self.mouse_mode = self._setup_mouse_mode()
        self._setup_toolbar()

        self._setup_ctxbar()

        self._connect()

    def _build_main_widget(self):
        self.widget = QMainWindow()
        self.widget.setWindowFlags(Qt.Tool)

        w = QWidget()
        l = QHBoxLayout()
        l.setSpacing(2)
        l.setContentsMargins(2, 2, 2, 2)
        w.setLayout(l)

        mpl = MplWidget()
        self.canvas = mpl.canvas
        l.addWidget(mpl)
        l.setStretchFactor(mpl, 5)

        self.widget.setCentralWidget(w)

    def _setup_ctxbar(self):
        l = self.widget.centralWidget().layout()
        self._contexts = [NavContext(self),
                          FitContext(self),
                          CollapseContext(self)]

        tabs = QTabWidget()
        tabs.addTab(self._contexts[0].widget, 'Navigate')
        tabs.addTab(self._contexts[1].widget, 'Fit')
        tabs.addTab(self._contexts[2].widget, 'Collapse')
        self._tabs = tabs
        self._tabs.setVisible(False)
        l.addWidget(tabs)
        l.setStretchFactor(tabs, 0)

    def _connect(self):
        add_callback(self.client, 'slice',
                     self._check_invalidate,
                     echo_old=True)

        def _on_tab_change(index):
            for i, ctx in enumerate(self._contexts):
                ctx.set_enabled(i == index)
                if i == index:
                    self.profile.active_grip = ctx.grip

        self._tabs.currentChanged.connect(_on_tab_change)
        _on_tab_change(self._tabs.currentIndex())

    def _setup_mouse_mode(self):
        # This will be added to the ImageWidget's toolbar
        mode = SpectrumExtractorMode(self.image_widget.client.axes,
                                     release_callback=self._update_profile)

        def toggle_mode_enabled(data):
            mode.enabled = (data.ndim > 2)

        add_callback(self.client, 'display_data', toggle_mode_enabled)
        return mode

    def _setup_toolbar(self):
        tb = GlueToolbar(self.canvas, self.widget)

        # disable ProfileViewer mouse processing during mouse modes
        tb.mode_activated.connect(self.profile.disconnect)
        tb.mode_deactivated.connect(self.profile.connect)

        self._menu_toggle_action = QAction("Options", tb)
        self._menu_toggle_action.setCheckable(True)
        self._menu_toggle_action.toggled.connect(self._toggle_menu)

        tb.addAction(self._menu_toggle_action)
        self.widget.addToolBar(tb)
        return tb

    def _toggle_menu(self, active):
        self._tabs.setVisible(active)

    def _check_invalidate(self, slc_old, slc_new):
        """
        If we change the orientation of the slice,
        reset and hide the profile viewer
        """
        if self.profile_axis is None:
            return

        if (slc_old.index('x') != slc_new.index('x') or
           slc_old.index('y') != slc_new.index('y')):
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

    def _recenter_grips(self):
        for ctx in self._contexts:
            ctx.recenter(self.axes.get_xlim())

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
        self.profile.set_xlabel(xlabel)
        self.profile.set_profile(x, y, c='k')

        # relim x range if requested
        if self._relim_requested:
            self._relim_requested = False
            self.axes.set_xlim(np.nanmin(x), np.nanmax(x))

        # relim y range to data within the view window
        self.profile.autoscale_ylim()

        if self.axes.get_xlim() != xlim:
            self._recenter_grips()

        self.axes.figure.canvas.draw()
        self.show()

    def _move_below_image_widget(self):
        rect = self.image_widget.frameGeometry()
        pos = self.image_widget.mapToGlobal(rect.bottomLeft())
        self.widget.setGeometry(pos.x(), pos.y(),
                                rect.width(), rect.width() / 1.5)

    def show(self):
        if self.widget.isVisible():
            return
        self._move_below_image_widget()
        self.widget.show()

    def hide(self):
        self.widget.close()
