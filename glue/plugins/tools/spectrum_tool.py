import traceback
import logging

import numpy as np

from ...external.qt.QtCore import Qt, Signal
from ...external.qt.QtGui import (QMainWindow, QWidget,
                                 QHBoxLayout, QTabWidget,
                                 QComboBox, QFormLayout, QPushButton,
                                 QAction, QTextEdit, QFont, QDialog,
                                 QDialogButtonBox, QLineEdit,
                                 QDoubleValidator, QCheckBox, QGridLayout,
                                 QLabel, QFileDialog)


from ...clients.profile_viewer import ProfileViewer
from ...qt.widgets.mpl_widget import MplWidget
from ...qt.mouse_mode import SpectrumExtractorMode
from ...core.callback_property import add_callback, ignore_callback
from ...core.util import Pointer
from ...core import Subset
from ...core.exceptions import IncompatibleAttribute
from ...qt.glue_toolbar import GlueToolbar
from ...qt.qtutil import load_ui, nonpartial, Worker
from ...qt.widget_properties import CurrentComboProperty
from ...core.aggregate import Aggregate
from ...qt.mime import LAYERS_MIME_TYPE
from ...qt.simpleforms import build_form_item
from ...config import fit_plugin
from ...external.six.moves import range as xrange
from ...qt.widgets.glue_mdi_area import GlueMdiSubWindow
from ...qt.decorators import messagebox_on_error
from ...utils import drop_axis


def setup():
    from ...logger import logger
    from ...config import tool_registry
    from ...qt.widgets import ImageWidget
    tool_registry.add(SpectrumTool, widget_cls=ImageWidget)
    logger.info("Loaded spectrum tool plugin")


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
    def spectrum(data, attribute, roi, slc, zaxis):
        xaxis = slc.index('x')
        yaxis = slc.index('y')
        ndim, nz = data.ndim, data.shape[zaxis]

        l, r, b, t = roi.xmin, roi.xmax, roi.ymin, roi.ymax
        shp = data.shape
        # The 'or 0' is because Numpy in Python 3 cannot deal with 'None'
        l, r = np.clip([l or 0, r or 0], 0, shp[xaxis])
        b, t = np.clip([b or 0, t or 0], 0, shp[yaxis])

        # extract sub-slice, without changing dimension
        slc = [slice(s, s + 1)
               if s not in ['x', 'y'] else slice(None)
               for s in slc]
        slc[xaxis] = slice(l, r)
        slc[yaxis] = slice(b, t)
        slc[zaxis] = slice(None)
        x = Extractor.abcissa(data, zaxis)

        data = data[attribute, tuple(slc)]
        finite = np.isfinite(data)

        assert data.ndim == ndim

        for i in reversed(list(range(ndim))):
            if i != zaxis:
                data = np.nansum(data, axis=i)
                finite = finite.sum(axis=i)

        assert data.ndim == 1
        assert data.size == nz

        data = (1. * data / finite).ravel()
        return x, data

    @staticmethod
    def world2pixel(data, axis, value):
        x = Extractor.abcissa(data, axis)
        if x.size > 1 and (x[1] < x[0]):
            x = x[::-1]
            result = x.size - np.searchsorted(x, value) - 2
        else:
            result = np.searchsorted(x, value) - 1
        return np.clip(result, 0, x.size - 1)

    @staticmethod
    def pixel2world(data, axis, value):
        x = Extractor.abcissa(data, axis)
        return x[np.clip(value, 0, x.size - 1)]

    @staticmethod
    def subset_spectrum(subset, attribute, slc, zaxis):
        """
        Extract a spectrum from a subset.

        This makes a mask of the subset in the **current slice**,
        and extracts a tube of this shape over all slices along ``zaxis``.
        In other words, the variation of the subset along ``zaxis`` is ignored,
        and only the interaction of the subset and the slice is relevant.

        :param subset: A :class:`~glue.core.subset.Subset`
        :param attribute: The :class:`~glue.core.data.ComponentID` to extract
        :param slc: A tuple describing the slice
        :param zaxis: Which axis to integrate over
        """
        data = subset.data
        x = Extractor.abcissa(data, zaxis)

        view = [slice(s, s + 1)
                if s not in ['x', 'y'] else slice(None)
                for s in slc]

        mask = np.squeeze(subset.to_mask(view))
        if slc.index('x') < slc.index('y'):
            mask = mask.T

        w = np.where(mask)
        view[slc.index('x')] = w[1]
        view[slc.index('y')] = w[0]

        result = np.empty(x.size)

        # treat each channel separately, to reduce memory storage
        for i in xrange(data.shape[zaxis]):
            view[zaxis] = i
            val = data[attribute, view]
            result[i] = np.nansum(val) / np.isfinite(val).sum()

        y = result

        return x, y


class SpectrumContext(object):

    """
    Base class for different interaction contexts
    """
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
        """ Create a :class:`~glue.clients.profile_viewer.Grip` object
            to interact with the plot. Assign to self.grip
        """
        raise NotImplementedError()

    def _setup_widget(self):
        """
        Create a context-specific widget
        """
        # this is the widget that is displayed to the right of the
        # spectrum
        raise NotImplementedError()

    def _connect(self):
        """
        Attach event handlers
        """
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

    """
    Mode to set the 2D slice in the parent image widget by dragging
    a handle in the spectrum
    """

    def _setup_grip(self):
        def _set_client_from_grip(value):
            """Update client.slice given grip value"""
            if not self.main.enabled:
                return

            slc = list(self.client.slice)
            # client.slice stored in pixel coords
            value = Extractor.world2pixel(
                self.data,
                self.profile_axis, value)
            slc[self.profile_axis] = value

            # prevent callback bouncing. Fixes #298
            with ignore_callback(self.grip, 'value'):
                self.client.slice = tuple(slc)

        def _set_grip_from_client(slc):
            """Update grip.value given client.slice"""
            if not self.main.enabled:
                return

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
        self.widget = QTextEdit()
        self.widget.setHtml("To <b> slide </b> through the cube, "
                            "drag the handle or double-click<br><br><br>"
                            "To make a <b> new profile </b>, "
                            "click-drag a new box in the image, or drag "
                            "a subset onto the plot to the left")
        self.widget.setTextInteractionFlags(Qt.NoTextInteraction)


class CollapseContext(SpectrumContext):

    """
    Mode to collapse a section of a cube into a 2D image.

    Supports several aggregations: mean, median, max, mom1, mom2
    """

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

        run = QPushButton("Collapse")
        save = QPushButton("Save as FITS file")

        buttons = QHBoxLayout()
        buttons.addWidget(run)
        buttons.addWidget(save)

        self._save = save
        self._run = run

        l.addRow("", combo)
        l.addRow("", buttons)

        self.widget = w
        self._combo = combo
        self._agg = None

    def _connect(self):
        self._run.clicked.connect(nonpartial(self._aggregate))
        self._save.clicked.connect(nonpartial(self._choose_save))

    @property
    def aggregator(self):
        return self._combo.itemData(self._combo.currentIndex())

    @property
    def aggregator_label(self):
        return self._combo.currentText()

    def _aggregate(self):
        func = self.aggregator

        rng = list(self.grip.range)
        rng[1] += 1
        rng = Extractor.world2pixel(self.data,
                                    self.profile_axis,
                                    rng)

        agg = Aggregate(self.data, self.client.display_attribute,
                        self.main.profile_axis, self.client.slice, rng)

        im = func(agg)
        self._agg = im
        self.client.override_image(im)

    @messagebox_on_error("Failed to export projection")
    def _choose_save(self):

        out, _ = QFileDialog.getSaveFileName(filter='FITS Files (*.fits)')
        if out is None:
            return

        self.save_to(out)

    def save_to(self, pth):
        """
        Write the projection to a file

        Parameters
        ----------
        pth : str
           Path to write to
        """
        from astropy.io import fits

        data = self.client.display_data
        if data is None:
            raise RuntimeError("Cannot save projection -- no data to visualize")

        self._aggregate()

        # try to project wcs to 2D
        wcs = getattr(data.coords, 'wcs', None)
        if wcs:
            try:
                wcs = drop_axis(wcs, data.ndim - 1 - self.main.profile_axis)
                header = wcs.to_header(True)
            except Exception as e:
                msg = "Could not extract 2D wcs for this data: %s" % e
                logging.getLogger(__name__).warn(msg)
                header = fits.Header()

        lo, hi = self.grip.range
        history = ('Created by Glue. %s projection over channels %i-%i of axis %i. Slice=%s' %
                   (self.aggregator_label, lo, hi, self.main.profile_axis, self.client.slice))

        header.add_history(history)

        fits.writeto(pth, self._agg, header, clobber=True)


class ConstraintsWidget(QWidget):

    """
    A widget to display and tweak the constraints of a :class:`~glue.core.fitters.BaseFitter1D`
    """

    def __init__(self, constraints, parent=None):
        """
        Parameters
        ----------
        constraints : dict
            The `contstraints` property of a :class:`~glue.core.fitters.BaseFitter1D`
            object
        parent : QWidget (optional)
            The parent of this widget
        """
        super(ConstraintsWidget, self).__init__(parent)
        self.constraints = constraints

        self.layout = QGridLayout()
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.layout.setSpacing(4)

        self.setLayout(self.layout)

        self.layout.addWidget(QLabel("Estimate"), 0, 1)
        self.layout.addWidget(QLabel("Fixed"), 0, 2)
        self.layout.addWidget(QLabel("Bounded"), 0, 3)
        self.layout.addWidget(QLabel("Lower Bound"), 0, 4)
        self.layout.addWidget(QLabel("Upper Bound"), 0, 5)

        self._widgets = {}
        names = sorted(list(self.constraints.keys()))

        for k in names:
            row = []
            w = QLabel(k)
            row.append(w)

            v = QDoubleValidator()
            e = QLineEdit()
            e.setValidator(v)
            e.setText(str(constraints[k]['value'] or ''))
            row.append(e)

            w = QCheckBox()
            w.setChecked(constraints[k]['fixed'])
            fix = w
            row.append(w)

            w = QCheckBox()
            limits = constraints[k]['limits']
            w.setChecked(limits is not None)
            bound = w
            row.append(w)

            e = QLineEdit()
            e.setValidator(v)
            if limits is not None:
                e.setText(str(limits[0]))
            row.append(e)

            e = QLineEdit()
            e.setValidator(v)
            if limits is not None:
                e.setText(str(limits[1]))
            row.append(e)

            def unset(w):
                def result(active):
                    if active:
                        w.setChecked(False)
                return result

            fix.toggled.connect(unset(bound))
            bound.toggled.connect(unset(fix))

            self._widgets[k] = row

        for i, row in enumerate(names, 1):
            for j, widget in enumerate(self._widgets[row]):
                self.layout.addWidget(widget, i, j)

    def settings(self, name):
        """ Return the constraints for a single model parameter """
        row = self._widgets[name]
        name, value, fixed, limited, lo, hi = row
        value = float(value.text()) if value.text() else None
        fixed = fixed.isChecked()
        limited = limited.isChecked()
        lo = lo.text()
        hi = hi.text()
        limited = limited and not ((not lo) or (not hi))
        limits = None if not limited else [float(lo), float(hi)]
        return dict(value=value, fixed=fixed, limits=limits)

    def update_constraints(self, fitter):
        """ Update the constraints in a :class:`~glue.core.fitters.BaseFitter1D`
            based on the settings in this widget
        """
        for name in self._widgets:
            s = self.settings(name)
            fitter.set_constraint(name, **s)


class FitSettingsWidget(QDialog):

    def __init__(self, fitter, parent=None):
        super(FitSettingsWidget, self).__init__(parent)
        self.fitter = fitter

        self._build_form()
        self._connect()
        self.setModal(True)

    def _build_form(self):
        fitter = self.fitter

        l = QFormLayout()
        options = fitter.options
        self.widgets = {}
        self.forms = {}

        for k in sorted(options):
            item = build_form_item(fitter, k)
            l.addRow(item.label, item.widget)
            self.widgets[k] = item.widget
            self.forms[k] = item  # need to prevent garbage collection

        constraints = fitter.constraints
        if constraints:
            self.constraints = ConstraintsWidget(constraints)
            l.addRow(self.constraints)
        else:
            self.constraints = None

        self.okcancel = QDialogButtonBox(QDialogButtonBox.Ok |
                                         QDialogButtonBox.Cancel)
        l.addRow(self.okcancel)
        self.setLayout(l)

    def _connect(self):
        self.okcancel.accepted.connect(self.accept)
        self.okcancel.rejected.connect(self.reject)
        self.accepted.connect(self.update_fitter_from_settings)

    def update_fitter_from_settings(self):
        for k, v in self.widgets.items():
            setattr(self.fitter, k, v.value())
        if self.constraints is not None:
            self.constraints.update_constraints(self.fitter)


class FitContext(SpectrumContext):

    """
    Mode to fit a range of a spectrum with a model fitter.

    Fitters are taken from user-defined fit plugins, or
    :class:`~glue.core.fitters.BaseFitter1D` subclasses
    """
    error = CurrentComboProperty('ui.uncertainty_combo')
    fitter = CurrentComboProperty('ui.profile_combo')

    def _setup_grip(self):
        self.grip = self.main.profile.new_range_grip()

    def _setup_widget(self):
        self.ui = load_ui('spectrum_fit_panel')
        self.ui.uncertainty_combo.hide()
        self.ui.uncertainty_label.hide()
        font = QFont("Courier")
        font.setStyleHint(font.Monospace)
        self.ui.results_box.document().setDefaultFont(font)
        self.ui.results_box.setLineWrapMode(self.ui.results_box.NoWrap)
        self.widget = self.ui

        for fitter in list(fit_plugin):
            self.ui.profile_combo.addItem(fitter.label,
                                          userData=fitter())

    def _edit_model_options(self):

        d = FitSettingsWidget(self.fitter)
        d.exec_()

    def _connect(self):
        self.ui.fit_button.clicked.connect(nonpartial(self.fit))
        self.ui.clear_button.clicked.connect(nonpartial(self.clear))
        self.ui.settings_button.clicked.connect(
            nonpartial(self._edit_model_options))

    def fit(self):
        """
        Fit a model to the data

        The fitting happens on a dedicated thread, to keep the UI
        responsive
        """
        xlim = self.grip.range
        fitter = self.fitter

        def on_success(result):
            fit_result, _, _, _ = result
            self._report_fit(fitter.summarize(*result))
            self.main.profile.plot_fit(fitter, fit_result)

        def on_fail(exc_info):
            exc = '\n'.join(traceback.format_exception(*exc_info))
            self._report_fit("Error during fitting:\n%s" % exc)

        def on_done():
            self.ui.fit_button.setText("Fit")
            self.ui.fit_button.setEnabled(True)
            self.canvas.draw()

        self.ui.fit_button.setText("Running...")
        self.ui.fit_button.setEnabled(False)

        w = Worker(self.main.profile.fit, fitter, xlim=xlim)
        w.result.connect(on_success)
        w.error.connect(on_fail)
        w.finished.connect(on_done)

        self._fit_worker = w  # hold onto a reference
        w.start()

    def _report_fit(self, report):
        self.ui.results_box.document().setPlainText(report)

    def clear(self):
        self.ui.results_box.document().setPlainText('')
        self.main.profile.clear_fit()
        self.canvas.draw()


class SpectrumMainWindow(QMainWindow):

    """
    The main window that the spectrum viewer is embedded in.

    Defines two signals to trigger when a subset is dropped into the window,
    and when the window is closed.
    """
    subset_dropped = Signal(object)
    window_closed = Signal()

    def __init__(self, parent=None):
        super(SpectrumMainWindow, self).__init__(parent=parent)
        self.setAcceptDrops(True)

    def closeEvent(self, event):
        self.window_closed.emit()
        return super(SpectrumMainWindow, self).closeEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(LAYERS_MIME_TYPE):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        layer = event.mimeData().data(LAYERS_MIME_TYPE)[0]
        if isinstance(layer, Subset):
            self.subset_dropped.emit(layer)


class SpectrumTool(object):

    """
    Main widget for interacting with spectra extracted from an image.

    Provides different contexts for interacting with the spectrum:

    *navigation context* lets the user set the slice in the parent image
                         by dragging a bar on the spectrum
    *fit context* lets the user fit models to a portion of the spectrum
    *collapse context* lets the users collapse a section of a cube to a 2D image
    """

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
        w = self.image_widget.session.application.add_widget(self,
                                                             label='Profile')
        w.close()

    def close(self):
        if hasattr(self, '_mdi_wrapper'):
            self._mdi_wrapper.close()
        else:
            self.widget.close()

    @property
    def enabled(self):
        """Return whether the window is visible and active"""
        return self.widget.isVisible()

    def mdi_wrap(self):
        sub = GlueMdiSubWindow()
        sub.setWidget(self.widget)
        self.widget.destroyed.connect(sub.close)
        sub.resize(self.widget.size())
        self._mdi_wrapper = sub
        return sub

    def _build_main_widget(self):
        self.widget = SpectrumMainWindow()
        self.widget.window_closed.connect(self.reset)

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

        self.widget.subset_dropped.connect(self._extract_subset_profile)

    def _setup_mouse_mode(self):
        # This will be added to the ImageWidget's toolbar
        mode = SpectrumExtractorMode(self.image_widget.client.axes,
                                     release_callback=self._update_profile)
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
        if self.profile_axis is None or not self.enabled:
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

    def _extract_subset_profile(self, subset):
        slc = self.client.slice
        try:
            x, y = Extractor.subset_spectrum(subset,
                                             self.client.display_attribute,
                                             slc,
                                             self.profile_axis)
        except IncompatibleAttribute:
            return

        self._set_profile(x, y)

    def _update_from_roi(self, roi):
        data = self.data
        att = self.client.display_attribute
        slc = self.client.slice

        if data is None or att is None:
            return

        zax = self.profile_axis

        x, y = Extractor.spectrum(data, att, roi, slc, zax)
        self._set_profile(x, y)

    def _update_profile(self, *args):
        roi = self.mouse_mode.roi()
        return self._update_from_roi(roi)

    def _set_profile(self, x, y):
        data = self.data

        xid = data.get_world_component_id(self.profile_axis)
        units = data.get_component(xid).units
        xlabel = str(xid) if units is None else '%s [%s]' % (xid, units)

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
        pos = rect.bottomLeft()
        self._mdi_wrapper.setGeometry(pos.x(), pos.y(),
                                      rect.width(), 300)

    def show(self):
        if self.widget.isVisible():
            return
        self._move_below_image_widget()
        self.widget.show()

    def hide(self):
        self.widget.close()

    def _get_modes(self, axes):
        return [self.mouse_mode]

    def _display_data_hook(self, data):
        if data is not None:
            self.mouse_mode.enabled = data.ndim > 2