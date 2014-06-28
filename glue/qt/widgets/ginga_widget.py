import logging
import sys, traceback
import numpy as np

from ...external.qt.QtGui import (QAction, QLabel, QCursor, QMainWindow,
                                  QToolButton, QToolBar, QIcon, QMessageBox,
                                  QActionGroup, QMdiSubWindow, QWidget,
                                  QVBoxLayout)

from ...external.qt.QtCore import Qt, QRect

from ginga.qtw.ImageViewCanvasQt import ImageViewCanvas
from ginga.qtw import Readout, ColorBar
from ginga.misc import log

from .data_viewer import DataViewer
from ... import core
from ... import config

from ...clients.ginga_client import GingaClient
from ...external.modest_image import imshow

from ...clients.layer_artist import Pointer
from ...core.callback_property import add_callback
from ...core import roi as roimod

from .data_slice_widget import DataSlice

from .mpl_widget import defer_draw

from ..decorators import set_cursor
from ..qtutil import cmap2pixmap, load_ui, get_icon, nonpartial
from ..widget_properties import CurrentComboProperty, ButtonProperty


__all__ = ['GingaWidget']


class GingaWidget(DataViewer):
    LABEL = "Ginga Viewer"
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
        super(GingaWidget, self).__init__(session, parent)

        #logger = logging.getLogger(__name__)
        #self.logger = log.get_logger(name='ginga', log_stderr=True)
        self.logger = log.get_logger(name='ginga', null=True)

        self.canvas = ImageViewCanvas(self.logger, render='widget')
        # prevent widget from grabbing focus 
        self.canvas.follow_focus = False
        # enable interactive features
        bindings = self.canvas.get_bindings()
        bindings.enable_all(True)
        self.canvas.set_callback('none-move', self.motion_readout)
        self.canvas.set_callback('draw-event', self._apply_roi_cb)
        self.canvas.set_callback('draw-down', self._clear_roi_cb)
        self.canvas.enable_draw(False)

        # Create settings and set defaults
        settings = self.canvas.get_settings()
        self.settings = settings
        settings.getSetting('cuts').add_callback('set', self.cut_levels_cb)
        settings.set(autozoom='override', autocuts='override',
                     autocenter=True)

        # make color bar, with color maps shared from ginga canvas
        self.colorbar = ColorBar.ColorBar(self.logger)
        rgbmap = self.canvas.get_rgbmap()
        rgbmap.add_callback('changed', self.rgbmap_cb, self.canvas)
        self.colorbar.set_rgbmap(rgbmap)

        # make coordinates/value readout
        self.readout = Readout.Readout(-1, -1)

        self.roi_tag = None

        # pack it all together in a bundle
        topw = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas.get_widget(), stretch=1)
        layout.addWidget(self.colorbar, stretch=0)
        layout.addWidget(self.readout.get_widget(), stretch=0)
        topw.setLayout(layout)

        self.central_widget = topw
        self.setCentralWidget(self.central_widget)
        self.ui = load_ui('imagewidget', None)
        self.option_widget = self.ui
        self.ui.slice = DataSlice()
        self.ui.slice_layout.addWidget(self.ui.slice)
        self.client = GingaClient(self._data,
                                  self.canvas,      # no mpl here (yet)
                                  artist_container=self._container)
        self._tweak_geometry()

        toolbar = self.make_toolbar()
        self.addToolBar(toolbar)
        self._connect()
        self._init_widgets()
        self.set_data(0)

        stbar = self.statusBar()
        stbar.setSizeGripEnabled(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._slice_widget = None

    def match_colorbar(self, canvas, colorbar):
        rgbmap = canvas.get_rgbmap()
        loval, hival = canvas.get_cut_levels()
        colorbar.set_range(loval, hival, redraw=False)
        colorbar.set_rgbmap(rgbmap)

    def rgbmap_cb(self, rgbmap, canvas):
        self.match_colorbar(canvas, self.colorbar)

    def cut_levels_cb(self, setting, tup):
        (loval, hival) = tup
        self.colorbar.set_range(loval, hival)

    def _tweak_geometry(self):
        self.central_widget.resize(600, 400)
        self.resize(self.central_widget.size())
        self.ui.rgb_options.hide()

    def make_toolbar(self):
        result = QToolBar(parent=self)
        agroup = QActionGroup(result)
        agroup.setExclusive(True)
        for (mode_text, mode_icon, mode_cb) in self._mouse_modes():
            # TODO: add icons similar to the Matplotlib toolbar
            action = result.addAction(mode_icon, mode_text, mode_cb)
            action.setCheckable(True)
            action = agroup.addAction(action)

        #cmap = _colormap_mode(self, self.client.set_cmap)
        #result.addWidget(cmap)
        return result

    def _mouse_modes(self):
        modes = []
        modes.append(("Rectangle", get_icon('glue_square'),
                      lambda: self._set_roi_mode('rectangle')))
        modes.append(("Circle", get_icon('glue_circle'),
                      lambda: self._set_roi_mode('circle')))
        modes.append(("Polygon", get_icon('glue_lasso'),
                      lambda: self._set_roi_mode('polygon')))
        return modes

    def _set_roi_mode(self, name):
        self.canvas.enable_draw(True)
        self.canvas.set_drawtype(name, color='cyan', linestyle='dash')
        print "ROI (%s) " % (name)

    def _clear_roi_cb(self, canvas, *args):
        print "roi cleared"
        try:
            self.canvas.deleteObjectByTag(self.roi_tag)
        except:
            pass

    def ginga_graphic_to_roi(self, obj):
        if obj.kind == 'rectangle':
            roi = roimod.RectangularROI(xmin=obj.x1, xmax=obj.x2,
                                        ymin=obj.y1, ymax=obj.y2)
        elif obj.kind == 'circle':
            roi = roimod.CircularROI(xc=obj.x, yc=obj.y,
                                     radius=obj.radius)
        elif obj.kind == 'polygon':
            vx = map(lambda xy: xy[0], obj.points)
            vy = map(lambda xy: xy[1], obj.points)
            roi = roimod.PolygonalROI(vx=vx, vy=vy)

        else:
            raise Exception("Don't know how to convert shape '%s' to a ROI" % (
                obj.kind))

        return roi
        
    def _apply_roi_cb(self, canvas, tag):
        self.roi_tag = tag
        obj = self.canvas.getObjectByTag(self.roi_tag)
        roi = self.ginga_graphic_to_roi(obj)
        print "ROI is", roi
        try:
            self.apply_roi(roi)
            # delete outline
            #self.canvas.deleteObjectByTag(self.roi_tag)
        except Exception as e:
            print "Error applying ROI: %s" % (str(e))
            (type, value, tb) = sys.exc_info()
            print "Traceback:\n%s" % ("".join(traceback.format_tb(tb)))
           
        
    def _extract_slice(self, roi):
        """
        Extract a PV-like slice, given a path traced on the widget
        """
        vx, vy = roi.to_polygon()
        pv, x, y = _slice_from_path(vx, vy, self.data, self.attribute, self.slice)
        if self._slice_widget is None:
            self._slice_widget = PVSliceWidget(pv, x, y, self)
            self._session.application.add_widget(self._slice_widget,
                                                 label='Custom Slice')
        else:
            self._slice_widget.set_image(pv, x, y)

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
        self.ui.slice.slice = value

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
        super(GingaWidget, self).register_to_hub(hub)
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
        for obj in [self, self.client]:
            hub.unsubscribe_all(obj)

    def remove_data_from_combo(self, data):
        """ Remvoe a data object from the combo box, if present """
        combo = self.ui.displayDataCombo
        pos = combo.findText(data.label)
        if pos >= 0:
            combo.removeItem(pos)

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
        return "Ginga Widget"

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
        super(GingaWidget, self).paintEvent(event)
        ## pos = self.central_widget.mapFromGlobal(QCursor.pos())
        ## x, y = pos.x(), self.central_widget.height() - pos.y()
        ## self._update_intensity_label(x, y)

    def motion_readout(self, canvas, button, data_x, data_y):

        # Get the value under the data coordinates
        try:
            #value = fitsimage.get_data(data_x, data_y)
            # We report the value across the pixel, even though the coords
            # change halfway across the pixel
            value = canvas.get_data(int(data_x+0.5), int(data_y+0.5))

        except Exception:
            value = None

        fits_x, fits_y = data_x + 1, data_y + 1

        # Calculate WCS RA
        try:
            # NOTE: image function operates on DATA space coords
            image = canvas.get_image()
            if image == None:
                # No image loaded
                return
            ra_txt, dec_txt = image.pixtoradec(fits_x, fits_y,
                                               format='str', coords='fits')
        except Exception as e:
            self.logger.warn("Bad coordinate conversion: %s" % (
                str(e)))
            ra_txt  = 'BAD WCS'
            dec_txt = 'BAD WCS'

        text = "RA: %s  DEC: %s  X: %.2f  Y: %.2f  Value: %s" % (
            ra_txt, dec_txt, fits_x, fits_y, value)
        self.readout.set_text(text)


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


class StandaloneGingaWidget(QMainWindow):

    """
    A simplified image viewer, without any brushing or linking,
    but with the ability to adjust contrast and resample.
    """

    def __init__(self, image, parent=None, **kwargs):
        """
        :param image: Image to display (2D numpy array)
        :param parent: Parent widget (optional)

        :param kwargs: Extra keywords to pass to imshow
        """
        super(StandaloneGingaWidget, self).__init__(parent)

        #logger = logging.getLogger(__name__)
        logger = log.get_logger(name='ginga', log_stderr=True)
        self.canvas = ImageViewCanvas(logger)
        self.central_widget = self.canvas.get_widget()
        self.setCentralWidget(self.central_widget)
        self._setup_axes()

        self._im = None

        #self.make_toolbar()
        self.set_image(image, **kwargs)

    def _setup_axes(self):
        ## self._axes = self.central_widget.canvas.fig.add_subplot(111)
        ## self._axes.set_aspect('equal', adjustable='datalim')
        pass

    def set_image(self, image, **kwargs):
        """
        Update the image shown in the widget
        """
        print "image is", image
        if self._im is not None:
            self._im.remove()
            self._im = None

        ## kwargs.setdefault('origin', 'upper')

        ## self._im = imshow(self._axes, image,
        ##                   norm=self._norm, cmap='gray', **kwargs)
        ## self._im_array = image
        ## self._axes.set_xticks([])
        ## self._axes.set_yticks([])
        self._redraw()

    @property
    def axes(self):
        """
        The Matplolib axes object for this figure
        """
        return self._axes

    def show(self):
        super(StandaloneGingaWidget, self).show()
        self._redraw()

    def _redraw(self):
        self.canvas.redraw()

    def _set_cmap(self, cmap):
        #self._im.set_cmap(cmap)
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

    def _set_norm(self, mode):
        """ Use the `ContrastMouseMode` to adjust the transfer function """
        clip_lo, clip_hi = mode.get_clip_percentile()
        stretch = mode.stretch
        ## self._norm.clip_lo = clip_lo
        ## self._norm.clip_hi = clip_hi
        ## self._norm.stretch = stretch
        ## self._norm.bias = mode.bias
        ## self._norm.contrast = mode.contrast
        ## self._im.set_norm(self._norm)
        print "loval, hival = %f,%f" % (clip_lo, clip_hi)
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


class PVSliceWidget(StandaloneGingaWidget):

    """ A standalone image widget with extra interactivity for PV slices """

    def __init__(self, image, x, y, image_widget):
        self._parent = image_widget
        super(PVSliceWidget, self).__init__(image, x=x, y=y)
        conn = self.axes.figure.canvas.mpl_connect
        self._down_id = conn('button_press_event', self._on_click)
        self._move_id = conn('motion_notify_event', self._on_move)

    def set_image(self, im, x, y):
        super(PVSliceWidget, self).set_image(im)
        self._axes.set_aspect('auto')
        self._axes.set_xlim(0, im.shape[1])
        self._axes.set_ylim(0, im.shape[0])
        self._slc = self._parent.slice
        self._x = x
        self._y = y

    def _sync_slice(self, event):
        s = list(self._slc)

        # XXX breaks if display_data changes
        _, _, z = self._pos_in_parent(event)
        s[_slice_index(self._parent.data, s)] = z
        self._parent.slice = tuple(s)

    def _draw_crosshairs(self, event):
        x, y, _ = self._pos_in_parent(event)
        ax = self._parent.client.axes
        m, = ax.plot([x], [y], '+', ms=12, mfc='none', mec='#de2d26',
                     mew=2, zorder=100)
        ax.figure.canvas.draw()
        m.remove()

    def _on_move(self, event):
        if not event.button:
            return

        if not event.inaxes or event.canvas.toolbar.mode != '':
            return

        self._sync_slice(event)
        self._draw_crosshairs(event)

    def _pos_in_parent(self, event):
        ind = np.clip(event.xdata, 0, self._im_array.shape[1] - 1)
        x = self._x[ind]
        y = self._y[ind]
        z = event.ydata

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

def cmap2pixmap(cmap, steps=50):
    """Convert a Ginga colormap into a QPixmap

    :param cmap: The colormap to use
    :type cmap: Ginga colormap instance (e.g. ginga.cmap.get_cmap('gray'))
    :param steps: The number of color steps in the output. Default=50
    :type steps: int

    :rtype: QPixmap
    """
    inds = np.linspace(0, 1, steps)
    n = len(cmap.clst) - 1
    tups = [ cmap.clst[int(x*n)] for x in inds ]
    rgbas = [QColor(int(r * 255), int(g * 255),
                    int(b * 255), 255).rgba() for r, g, b in tups]
    im = QImage(steps, 1, QImage.Format_Indexed8)
    im.setColorTable(rgbas)
    for i in range(steps):
        im.setPixel(i, 0, i)
    im = im.scaled(128, 32)
    pm = QPixmap.fromImage(im)
    return pm
