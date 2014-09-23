from __future__ import absolute_import, division, print_function

import numpy as np

from ...external.qt.QtGui import (QAction, QLabel, QCursor, QMainWindow,
                                  QToolButton, QIcon, QMessageBox,
                                  QMdiSubWindow)

from ...external.qt.QtCore import Qt, QRect, Signal

from .data_viewer import DataViewer
from ... import core
from ... import config

from ...clients.image_client import ImageClient
from ...clients.ds9norm import DS9Normalize
from ...external.modest_image import imshow

from ...clients.layer_artist import Pointer
from ...core.callback_property import add_callback

from .data_slice_widget import DataSlice

from ..mouse_mode import (RectangleMode, CircleMode, PolyMode,
                          ContrastMode, PathMode)
from ..glue_toolbar import GlueToolbar
from ..spectrum_tool import SpectrumTool
from .mpl_widget import MplWidget, defer_draw

from ..qtutil import cmap2pixmap, load_ui, get_icon, nonpartial
from ..widget_properties import CurrentComboProperty, ButtonProperty

WARN_THRESH = 10000000  # warn when contouring large images

__all__ = ['ImageWidget']


class ImageWidget(DataViewer):
    LABEL = "Image Viewer"
    _property_set = DataViewer._property_set + \
        'data attribute rgb_mode rgb_viz ratt gatt batt slice'.split()

    attribute = CurrentComboProperty('ui.attributeComboBox',
                                     'Current attribute')
    data = CurrentComboProperty('ui.displayDataCombo',
                                'Current data')
    rgb_mode = ButtonProperty('ui.rgb',
                              'RGB Mode?')
    rgb_viz = Pointer('ui.rgb_options.rgb_visible')

    def __init__(self, session, parent=None):
        super(ImageWidget, self).__init__(session, parent)
        self.central_widget = MplWidget()
        self.label_widget = QLabel("", self.central_widget)
        self.setCentralWidget(self.central_widget)
        self.ui = load_ui('imagewidget', None)
        self.option_widget = self.ui
        self.ui.slice = DataSlice()
        self.ui.slice_layout.addWidget(self.ui.slice)
        self.client = ImageClient(self._data,
                                  self.central_widget.canvas.fig,
                                  artist_container=self._container)
        self._spectrum_tool = SpectrumTool(self)
        self._tweak_geometry()

        self.make_toolbar()
        self._connect()
        self._init_widgets()
        self.set_data(0)
        self.statusBar().setSizeGripEnabled(False)
        self.setFocusPolicy(Qt.StrongFocus)
        self._slice_widget = None

    def _tweak_geometry(self):
        self.central_widget.resize(600, 400)
        self.resize(self.central_widget.size())
        self.ui.rgb_options.hide()

    def make_toolbar(self):
        result = GlueToolbar(self.central_widget.canvas, self, name='Image')
        for mode in self._mouse_modes():
            result.add_mode(mode)

        cmap = _colormap_mode(self, self.client.set_cmap)
        result.addWidget(cmap)

        # connect viewport update buttons to client commands to
        # allow resampling
        cl = self.client
        result.buttons['HOME'].triggered.connect(nonpartial(cl.check_update))
        result.buttons['FORWARD'].triggered.connect(nonpartial(
            cl.check_update))
        result.buttons['BACK'].triggered.connect(nonpartial(cl.check_update))

        self.addToolBar(result)
        return result

    def _mouse_modes(self):
        axes = self.client.axes

        def apply_mode(mode):
            self.apply_roi(mode.roi())

        def slice(mode):
            self._extract_slice(mode.roi())

        rect = RectangleMode(axes, roi_callback=apply_mode)
        circ = CircleMode(axes, roi_callback=apply_mode)
        poly = PolyMode(axes, roi_callback=apply_mode)
        contrast = ContrastMode(axes, move_callback=self._set_norm)
        spectrum = self._spectrum_tool.mouse_mode
        path = PathMode(axes, roi_callback=slice)

        def toggle_3d_modes(data):
            if data is None:
                return
            is3d = data.ndim > 2
            path.enabled = is3d
            spectrum.enabled = is3d

        add_callback(self.client, 'display_data', toggle_3d_modes)

        self._contrast = contrast
        self._path = path
        return [rect, circ, poly, contrast, spectrum, path]

    def _extract_slice(self, roi):
        """
        Extract a PV-like slice, given a path traced on the widget
        """
        vx, vy = roi.to_polygon()
        pv, x, y = _slice_from_path(vx, vy, self.data, self.attribute, self.slice)
        if self._slice_widget is None:
            self._slice_widget = PVSliceWidget(pv, x, y, self,
                                               interpolation='nearest')
            self._session.application.add_widget(self._slice_widget,
                                                 label='Custom Slice')
            self._slice_widget.window_closed.connect(self._path.clear)
        else:
            self._slice_widget.set_image(pv, x, y, interpolation='nearest')

        result = self._slice_widget
        result.axes.set_xlabel("Position Along Slice")
        result.axes.set_ylabel(_slice_label(self.data, self.slice))

        result.show()

    def _init_widgets(self):
        pass

    @defer_draw
    def add_data(self, data):
        """Private method to ingest new data into widget"""
        self.client.add_layer(data)
        self.add_data_to_combo(data)
        self.set_data(self._data_index(data))
        return True

    @defer_draw
    def add_subset(self, subset):
        self.client.add_scatter_layer(subset)
        assert subset in self.client.artists

    def _data_index(self, data):
        combo = self.ui.displayDataCombo

        for i in range(combo.count()):
            if combo.itemData(i) is data:
                return i

        return None

    def add_data_to_combo(self, data):
        """ Add a data object to the combo box, if not already present
        """
        if not self.client.can_image_data(data):
            return
        combo = self.ui.displayDataCombo
        label = data.label
        pos = combo.findText(label)
        if pos == -1:
            combo.addItem(label, userData=data)
        assert combo.findText(label) >= 0

    @property
    def ratt(self):
        """ComponentID assigned to R channel in RGB Mode"""
        return self.ui.rgb_options.attributes[0]

    @ratt.setter
    def ratt(self, value):
        att = list(self.ui.rgb_options.attributes)
        att[0] = value
        self.ui.rgb_options.attributes = att

    @property
    def gatt(self):
        """ComponentID assigned to G channel in RGB Mode"""
        return self.ui.rgb_options.attributes[1]

    @gatt.setter
    def gatt(self, value):
        att = list(self.ui.rgb_options.attributes)
        att[1] = value
        self.ui.rgb_options.attributes = att

    @property
    def batt(self):
        """ComponentID assigned to B channel in RGB Mode"""
        return self.ui.rgb_options.attributes[2]

    @batt.setter
    def batt(self, value):
        att = list(self.ui.rgb_options.attributes)
        att[2] = value
        self.ui.rgb_options.attributes = att

    @defer_draw
    def set_data(self, index):
        if index is None:
            return

        if self.ui.displayDataCombo.count() == 0:
            return

        data = self.ui.displayDataCombo.itemData(index)
        self.ui.slice.set_data(data)
        self.client.set_data(data)
        self.client.slice = self.ui.slice.slice
        self.ui.displayDataCombo.setCurrentIndex(index)
        self.set_attribute_combo(data)
        self._update_window_title()

    @property
    def slice(self):
        return self.client.slice

    @slice.setter
    def slice(self, value):
        self.client.slice = value

    @defer_draw
    def set_attribute(self, index):
        combo = self.ui.attributeComboBox
        component_id = combo.itemData(index)
        self.client.set_attribute(component_id)
        self.ui.attributeComboBox.setCurrentIndex(index)
        self._update_window_title()

    def set_attribute_combo(self, data):
        """ Update attribute combo box to reflect components in data"""
        combo = self.ui.attributeComboBox
        combo.blockSignals(True)
        combo.clear()
        fields = data.visible_components
        index = 0
        for i, f in enumerate(fields):
            combo.addItem(f.label, userData=f)
            if f == self.client.display_attribute:
                index = i
        combo.blockSignals(False)
        combo.setCurrentIndex(index)
        self.set_attribute(index)

    def _connect(self):
        ui = self.ui

        ui.displayDataCombo.currentIndexChanged.connect(self.set_data)
        ui.attributeComboBox.currentIndexChanged.connect(self.set_attribute)

        ui.monochrome.toggled.connect(self._update_rgb_console)
        ui.rgb_options.colors_changed.connect(self._update_window_title)
        ui.rgb_options.current_changed.connect(
            lambda: self._toolbars[0].set_mode(self._contrast))
        ui.slice.slice_changed.connect(self._update_slice)

        update_ui_slice = lambda val: setattr(ui.slice, 'slice', val)
        add_callback(self.client, 'slice', update_ui_slice)

    @defer_draw
    def _update_slice(self):
        self.client.slice = self.ui.slice.slice

    @defer_draw
    def _update_rgb_console(self, is_monochrome):
        if is_monochrome:
            self.ui.rgb_options.hide()
            self.ui.mono_att_label.show()
            self.ui.attributeComboBox.show()
            self.client.rgb_mode(False)
        else:
            self.ui.mono_att_label.hide()
            self.ui.attributeComboBox.hide()
            self.ui.rgb_options.show()
            rgb = self.client.rgb_mode(True)
            if rgb is not None:
                self.ui.rgb_options.artist = rgb

    def register_to_hub(self, hub):
        super(ImageWidget, self).register_to_hub(hub)
        self.client.register_to_hub(hub)

        dc_filt = lambda x: x.sender is self.client._data
        layer_present_filter = lambda x: x.data in self.client.artists

        hub.subscribe(self,
                      core.message.DataCollectionAddMessage,
                      handler=lambda x: self.add_data_to_combo(x.data),
                      filter=dc_filt)
        hub.subscribe(self,
                      core.message.DataCollectionDeleteMessage,
                      handler=lambda x: self.remove_data_from_combo(x.data),
                      filter=dc_filt)
        hub.subscribe(self,
                      core.message.DataUpdateMessage,
                      handler=lambda x: self._sync_data_labels()
                      )
        hub.subscribe(self,
                      core.message.ComponentsChangedMessage,
                      handler=lambda x: self.set_attribute_combo(x.data),
                      filter=layer_present_filter)

    def unregister(self, hub):
        super(ImageWidget, self).unregister(hub)
        for obj in [self, self.client]:
            hub.unsubscribe_all(obj)

    def remove_data_from_combo(self, data):
        """ Remvoe a data object from the combo box, if present """
        combo = self.ui.displayDataCombo
        pos = combo.findText(data.label)
        if pos >= 0:
            combo.removeItem(pos)

    def _set_norm(self, mode):
        """ Use the `ContrastMouseMode` to adjust the transfer function """
        clip_lo, clip_hi = mode.get_clip_percentile()
        stretch = mode.stretch
        return self.client.set_norm(clip_lo=clip_lo, clip_hi=clip_hi,
                                    stretch=stretch,
                                    bias=mode.bias, contrast=mode.contrast)

    def _update_window_title(self):
        if self.client.display_data is None:
            title = ''
        else:
            data = self.client.display_data.label
            a = self.client.rgb_mode()
            if a is None:  # monochrome mode
                title = "%s - %s" % (self.client.display_data.label,
                                     self.client.display_attribute.label)
            else:
                r = a.r.label if a.r is not None else ''
                g = a.g.label if a.g is not None else ''
                b = a.b.label if a.b is not None else ''
                title = "%s Red = %s  Green = %s  Blue = %s" % (data, r, g, b)
        self.setWindowTitle(title)

    def _update_data_combo(self):
        combo = self.ui.displayDataCombo
        for i in range(combo.count()):
            combo.setItemText(i, combo.itemData(i).label)

    def _sync_data_labels(self):
        self._update_window_title()
        self._update_data_combo()

    def __str__(self):
        return "Image Widget"

    def _confirm_large_image(self, data):
        """Ask user to confirm expensive operations

        :rtype: bool. Whether the user wishes to continue
        """

        warn_msg = ("WARNING: Image has %i pixels, and may render slowly."
                    " Continue?" % data.size)
        title = "Contour large image?"
        ok = QMessageBox.Ok
        cancel = QMessageBox.Cancel
        buttons = ok | cancel
        result = QMessageBox.question(self, title, warn_msg,
                                      buttons=buttons,
                                      defaultButton=cancel)
        return result == ok

    def options_widget(self):
        return self.option_widget

    @defer_draw
    def restore_layers(self, rec, context):
        self.client.restore_layers(rec, context)
        for artist in self.layers:
            self.add_data_to_combo(artist.layer.data)

        self.set_attribute_combo(self.client.display_data)
        self._update_data_combo()

    def paintEvent(self, event):
        super(ImageWidget, self).paintEvent(event)
        pos = self.central_widget.canvas.mapFromGlobal(QCursor.pos())
        x, y = pos.x(), self.central_widget.canvas.height() - pos.y()
        self._update_intensity_label(x, y)

    def _intensity_label(self, x, y):
        x, y = self.client.axes.transData.inverted().transform([x, y])
        value = self.client.point_details(x, y)['value']
        lbl = '' if value is None else "data: %s" % value
        return lbl

    def _update_intensity_label(self, x, y):
        lbl = self._intensity_label(x, y)
        self.label_widget.setText(lbl)

        fm = self.label_widget.fontMetrics()
        w, h = fm.width(lbl), fm.height()
        g = QRect(20, self.central_widget.geometry().height() - h, w, h)
        self.label_widget.setGeometry(g)


class ColormapAction(QAction):

    def __init__(self, label, cmap, parent):
        super(ColormapAction, self).__init__(label, parent)
        self.cmap = cmap
        pm = cmap2pixmap(cmap)
        self.setIcon(QIcon(pm))


def _colormap_mode(parent, on_trigger):

    # actions for each colormap
    acts = []
    for label, cmap in config.colormaps:
        a = ColormapAction(label, cmap, parent)
        a.triggered.connect(nonpartial(on_trigger, cmap))
        acts.append(a)

    # Toolbar button
    tb = QToolButton()
    tb.setWhatsThis("Set color scale")
    tb.setToolTip("Set color scale")
    icon = get_icon('glue_rainbow')
    tb.setIcon(icon)
    tb.setPopupMode(QToolButton.InstantPopup)
    tb.addActions(acts)

    return tb


class StandaloneImageWidget(QMainWindow):

    """
    A simplified image viewer, without any brushing or linking,
    but with the ability to adjust contrast and resample.
    """
    window_closed = Signal()

    def __init__(self, image, parent=None, **kwargs):
        """
        :param image: Image to display (2D numpy array)
        :param parent: Parent widget (optional)

        :param kwargs: Extra keywords to pass to imshow
        """
        super(StandaloneImageWidget, self).__init__(parent)
        self.central_widget = MplWidget()
        self.setCentralWidget(self.central_widget)
        self._setup_axes()

        self._im = None
        self._norm = DS9Normalize()

        self.make_toolbar()
        self.set_image(image, **kwargs)

    def _setup_axes(self):
        self._axes = self.central_widget.canvas.fig.add_subplot(111)
        self._axes.set_aspect('equal', adjustable='datalim')

    def set_image(self, image, **kwargs):
        """
        Update the image shown in the widget
        """
        if self._im is not None:
            self._im.remove()
            self._im = None

        kwargs.setdefault('origin', 'upper')

        self._im = imshow(self._axes, image,
                          norm=self._norm, cmap='gray', **kwargs)
        self._im_array = image
        self._axes.set_xticks([])
        self._axes.set_yticks([])
        self._redraw()

    @property
    def axes(self):
        """
        The Matplolib axes object for this figure
        """
        return self._axes

    def show(self):
        super(StandaloneImageWidget, self).show()
        self._redraw()

    def _redraw(self):
        self.central_widget.canvas.draw()

    def _set_cmap(self, cmap):
        self._im.set_cmap(cmap)
        self._redraw()

    def mdi_wrap(self):
        """
        Embed this widget in a QMdiSubWindow
        """
        sub = QMdiSubWindow()
        sub.setWidget(self)
        self.destroyed.connect(sub.close)
        sub.resize(self.size())
        self._mdi_wrapper = sub

        return sub

    def closeEvent(self, event):
        self.window_closed.emit()
        return super(StandaloneImageWidget, self).closeEvent(event)

    def _set_norm(self, mode):
        """ Use the `ContrastMouseMode` to adjust the transfer function """
        clip_lo, clip_hi = mode.get_clip_percentile()
        stretch = mode.stretch
        self._norm.clip_lo = clip_lo
        self._norm.clip_hi = clip_hi
        self._norm.stretch = stretch
        self._norm.bias = mode.bias
        self._norm.contrast = mode.contrast
        self._im.set_norm(self._norm)
        self._redraw()

    def make_toolbar(self):
        """
        Setup the toolbar
        """
        result = GlueToolbar(self.central_widget.canvas, self,
                             name='Image')
        result.add_mode(ContrastMode(self._axes, move_callback=self._set_norm))
        cm = _colormap_mode(self, self._set_cmap)
        result.addWidget(cm)
        self._cmap_actions = cm.actions()
        self.addToolBar(result)
        return result


class PVSliceWidget(StandaloneImageWidget):

    """ A standalone image widget with extra interactivity for PV slices """

    def __init__(self, image, x, y, image_widget, **kwargs):
        """
        :param image: 2D Numpy array representing the PV Slice
        :param x: X coordinate for each horizontal position
        :param y: Y coordinate for each horizontal position
        :param image_widget: Parent widget this was extracted from
        :param kwargs: Extra keywords are passed to imshow
        """
        self._crosshairs = None
        self._parent = image_widget
        super(PVSliceWidget, self).__init__(image, x=x, y=y, **kwargs)
        conn = self.axes.figure.canvas.mpl_connect
        self._down_id = conn('button_press_event', self._on_click)
        self._move_id = conn('motion_notify_event', self._on_move)
        self.axes.format_coord = self._format_coord

    def _format_coord(self, x, y):
        """
        Return a formatted location label for the taskbar

        :param x: x pixel location in slice array
        :param y: y pixel location in slice array
        """
        # xy -> xyz in image view
        pix = self._pos_in_parent(xdata=x, ydata=y)

        # xyz -> data pixel coords
        # accounts for fact that image might be shown transpoed/rotated
        s = list(self._slc)
        idx = _slice_index(self._parent.data, self._slc)
        s[s.index('x')] = pix[0]
        s[s.index('y')] = pix[1]
        s[idx] = pix[2]

        labels = self._parent.client.coordinate_labels(s)
        return '         '.join(labels)

    def set_image(self, im, x, y, **kwargs):
        super(PVSliceWidget, self).set_image(im, **kwargs)
        self._axes.set_aspect('auto')
        self._axes.set_xlim(0, im.shape[1])
        self._axes.set_ylim(0, im.shape[0])
        self._slc = self._parent.slice
        self._x = x
        self._y = y

    @defer_draw
    def _sync_slice(self, event):
        s = list(self._slc)
        # XXX breaks if display_data changes
        _, _, z = self._pos_in_parent(event)
        s[_slice_index(self._parent.data, s)] = z
        self._parent.slice = tuple(s)

    @defer_draw
    def _draw_crosshairs(self, event):
        if self._crosshairs is not None:
            self._crosshairs.remove()

        x, y, _ = self._pos_in_parent(event)
        ax = self._parent.client.axes
        self._crosshairs, = ax.plot([x], [y], '+', ms=12,
                                    mfc='none', mec='#de2d26',
                                    mew=2, zorder=100)
        ax.figure.canvas.draw()

    @defer_draw
    def _on_move(self, event):
        if not event.button:
            return

        if not event.inaxes or event.canvas.toolbar.mode != '':
            return

        self._sync_slice(event)
        self._draw_crosshairs(event)

    def _pos_in_parent(self, event=None, xdata=None, ydata=None):
        if event is not None:
            xdata = event.xdata
            ydata = event.ydata
        ind = np.clip(xdata, 0, self._im_array.shape[1] - 1)
        x = self._x[ind]
        y = self._y[ind]
        z = ydata

        return x, y, z

    def _on_click(self, event):
        if not event.inaxes or event.canvas.toolbar.mode != '':
            return
        self._sync_slice(event)
        self._draw_crosshairs(event)


def _slice_index(data, slc):
    """
    The axis over which to extract PV slices
    """
    return max([i for i in range(len(slc))
               if isinstance(slc[i], int)],
               key=lambda x: data.shape[x])


def _slice_from_path(x, y, data, attribute, slc):
    """
    Extract a PV-like slice from a cube

    :param x: An array of x values to extract (pixel units)
    :param y: An array of y values to extract (pixel units)
    :param data: :class:`~glue.core.data.Data`
    :param attribute: :claass:`~glue.core.data.Component`
    :param slc: orientation of the image widget that `pts` are defined on

    :returns: (slice, x, y)
              slice is a 2D Numpy array, corresponding to a "PV ribbon"
              cutout from the cube
              x and y are the resampled points along which the
              ribbon is extracted

    :note: For >3D cubes, the "V-axis" of the PV slice is the longest
           cube axis ignoring the x/y axes of `slc`
    """
    from ...external.pvextractor import Path, extract_pv_slice
    p = Path(list(zip(x, y)))

    cube = data[attribute]
    dims = list(range(data.ndim))
    s = list(slc)
    ind = _slice_index(data, slc)

    # transpose cube to (z, y, x, <whatever>)
    def _swap(x, s, i, j):
        x[i], x[j] = x[j], x[i]
        s[i], s[j] = s[j], s[i]

    _swap(dims, s, ind, 0)
    _swap(dims, s, s.index('y'), 1)
    _swap(dims, s, s.index('x'), 2)
    cube = cube.transpose(dims)

    # slice down from >3D to 3D if needed
    s = [slice(None)] * 3 + [slc[d] for d in dims[3:]]
    cube = cube[s]

    # sample cube
    spacing = 1  # pixel
    x, y = [np.round(_x).astype(int) for _x in p.sample_points(spacing)]
    result = extract_pv_slice(cube, p, order=0).data

    return result, x, y


def _slice_label(data, slc):
    """
    Returns a formatted axis label corresponding to the slice dimension
    in a PV slice

    :param data: Data that slice is extracted from
    :param slc: orientation in the image widget from which the PV slice
                was defined
    """
    idx = _slice_index(data, slc)
    return data.get_world_component_id(idx).label
