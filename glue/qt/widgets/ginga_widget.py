import logging
import sys
import traceback
import os.path
import numpy as np

from ...external.qt.QtGui import (QAction, QMainWindow,
                                  QToolButton, QToolBar, QIcon,
                                  QActionGroup, QMdiSubWindow, QWidget,
                                  QVBoxLayout, QColor, QImage, QPixmap)

from ...external.qt.QtCore import Qt, QRect, QSize

from ginga.qtw.ImageViewCanvasQt import ImageViewCanvas
from ginga.qtw import Readout, ColorBar
from ginga.misc import log
from ginga import cmap as ginga_cmap
# ginga_cmap.add_matplotlib_cmaps()

from .image_widget import ImageWidgetBase
from ... import config

from ...clients.ginga_client import GingaClient

from ...core import roi as roimod

from ..qtutil import cmap2pixmap, get_icon, nonpartial


from ..decorators import set_cursor
from ..qtutil import cmap2pixmap, load_ui, get_icon, nonpartial
from ..widget_properties import CurrentComboProperty, ButtonProperty

# Find out location of ginga module so we can some of its icons
ginga_home = os.path.split(sys.modules['ginga'].__file__)[0]
ginga_icon_dir = os.path.join(ginga_home, 'icons')

__all__ = ['GingaWidget']


class GingaWidget(ImageWidgetBase):

    LABEL = "Ginga Viewer"

    def __init__(self, session, parent=None):

        #logger = logging.getLogger(__name__)
        self.logger = log.get_logger(name='ginga', log_stderr=True)
        #self.logger = log.get_logger(name='ginga', null=True)

        self.canvas = ImageViewCanvas(self.logger, render='widget')

        # prevent widget from grabbing focus
        self.canvas.set_follow_focus(False)
        self.canvas.enable_overlays(True)

        # enable interactive features
        bindings = self.canvas.get_bindings()
        bindings.enable_all(True)
        self.canvas.set_callback('none-move', self.motion_readout)
        self.canvas.set_callback('draw-event', self._apply_roi_cb)
        self.canvas.set_callback('draw-down', self._clear_roi_cb)
        self.canvas.enable_draw(False)
        self.canvas.enable_autozoom('off')
        self.canvas.set_zoom_algorithm('rate')
        self.canvas.set_zoomrate(1.4)

        bm = self.canvas.get_bindmap()
        bm.add_callback('mode-set', self.mode_set_cb)
        self.mode_w = None
        self.mode_actns = {}

        # Create settings and set defaults
        settings = self.canvas.get_settings()
        self.settings = settings
        settings.getSetting('cuts').add_callback('set', self.cut_levels_cb)
        settings.set(autozoom='off', autocuts='override',
                     autocenter='override')

        # make color bar, with color maps shared from ginga canvas
        self.colorbar = ColorBar.ColorBar(self.logger)
        rgbmap = self.canvas.get_rgbmap()
        rgbmap.add_callback('changed', self.rgbmap_cb, self.canvas)
        self.colorbar.set_rgbmap(rgbmap)

        # make coordinates/value readout
        self.readout = Readout.Readout(-1, -1)
        self.roi_tag = None

        super(GingaWidget, self).__init__(session, parent)

    def make_client(self):
        return GingaClient(self._data, self.canvas, self._container)

    def make_central_widget(self):

        topw = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas.get_widget(), stretch=1)
        layout.addWidget(self.colorbar, stretch=0)
        layout.addWidget(self.readout.get_widget(), stretch=0)
        topw.setLayout(layout)
        return topw

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

    def make_toolbar(self):
        tb = QToolBar(parent=self)
        tb.setIconSize(QSize(25, 25))
        tb.layout().setSpacing(1)
        tb.setFocusPolicy(Qt.StrongFocus)

        agroup = QActionGroup(tb)
        agroup.setExclusive(True)
        for (mode_text, mode_icon, mode_cb) in self._mouse_modes():
            # TODO: add icons similar to the Matplotlib toolbar
            action = tb.addAction(mode_icon, mode_text)
            action.setCheckable(True)
            action.toggled.connect(mode_cb)
            agroup.addAction(action)

        action = tb.addAction(get_icon('glue_move'), "Pan")
        self.mode_actns['pan'] = action
        action.setCheckable(True)
        action.toggled.connect(lambda tf: self.mode_cb('pan', tf))
        agroup.addAction(action)
        icon = QIcon(os.path.join(ginga_icon_dir, 'hand_48.png'))
        action = tb.addAction(icon, "Free Pan")
        self.mode_actns['freepan'] = action
        action.setCheckable(True)
        action.toggled.connect(lambda tf: self.mode_cb('freepan', tf))
        agroup.addAction(action)
        icon = QIcon(os.path.join(ginga_icon_dir, 'rotate_48.png'))
        action = tb.addAction(icon, "Rotate")
        self.mode_actns['rotate'] = action
        action.setCheckable(True)
        action.toggled.connect(lambda tf: self.mode_cb('rotate', tf))
        agroup.addAction(action)
        action = tb.addAction(get_icon('glue_contrast'), "Contrast")
        self.mode_actns['contrast'] = action
        action.setCheckable(True)
        action.toggled.connect(lambda tf: self.mode_cb('contrast', tf))
        agroup.addAction(action)
        icon = QIcon(os.path.join(ginga_icon_dir, 'cuts_48.png'))
        action = tb.addAction(icon, "Cuts")
        self.mode_actns['cuts'] = action
        action.setCheckable(True)
        action.toggled.connect(lambda tf: self.mode_cb('cuts', tf))
        agroup.addAction(action)

        cmap_w = _colormap_mode(self, self.client.set_cmap)
        tb.addWidget(cmap_w)
        return tb

    def _mouse_modes(self):
        modes = []
        modes.append(("Rectangle", get_icon('glue_square'),
                      lambda tf: self._set_roi_mode('rectangle', tf)))
        modes.append(("Circle", get_icon('glue_circle'),
                      lambda tf: self._set_roi_mode('circle', tf)))
        modes.append(("Polygon", get_icon('glue_lasso'),
                      lambda tf: self._set_roi_mode('polygon', tf)))
        return modes

    def _set_roi_mode(self, name, tf):
        self.canvas.enable_draw(True)
        self.canvas.set_drawtype(name, color='cyan', linestyle='dash')
        bm = self.canvas.get_bindmap()
        bm.set_modifier('draw', modtype='locked')

    def _clear_roi_cb(self, canvas, *args):
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
        # delete outline
        self.canvas.deleteObject(obj, redraw=False)
        try:
            self.apply_roi(roi)
        except Exception as e:
            (type, value, tb) = sys.exc_info()
            print "Traceback:\n%s" % ("".join(traceback.format_tb(tb)))

    def _init_widgets(self):
        pass

    def motion_readout(self, canvas, button, data_x, data_y):

        # Get the value under the data coordinates
        try:
            #value = fitsimage.get_data(data_x, data_y)
            # We report the value across the pixel, even though the coords
            # change halfway across the pixel
            value = canvas.get_data(int(data_x + 0.5), int(data_y + 0.5))

        except Exception:
            value = None

        fits_x, fits_y = data_x + 1, data_y + 1

        # Calculate WCS RA
        try:
            # NOTE: image function operates on DATA space coords
            image = canvas.get_image()
            if image is None:
                # No image loaded
                return
            ra_txt, dec_txt = image.pixtoradec(fits_x, fits_y,
                                               format='str', coords='fits')
        except Exception as e:
            self.logger.warn("Bad coordinate conversion: %s" % (
                str(e)))
            ra_txt = 'BAD WCS'
            dec_txt = 'BAD WCS'

        text = "RA: %s  DEC: %s  X: %.2f  Y: %.2f  Value: %s" % (
            ra_txt, dec_txt, fits_x, fits_y, value)
        self.readout.set_text(text)

    def motion_readout(self, canvas, button, data_x, data_y):
        """This method is called when the user moves the mouse around the Ginga
        canvas.
        """

        d = self.client.point_details(data_x, data_y)

        # Get the value under the data coordinates
        try:
            #value = fitsimage.get_data(data_x, data_y)
            # We report the value across the pixel, even though the coords
            # change halfway across the pixel
            value = canvas.get_data(int(data_x + 0.5), int(data_y + 0.5))

        except Exception:
            value = None

        x_lbl, y_lbl = d['labels'][0], d['labels'][1]
        #x_txt, y_txt = d['world'][0], d['world'][1]

        text = "%s  %s  X=%.2f  Y=%.2f  Value=%s" % (
            x_lbl, y_lbl, data_x, data_y, value)
        self.readout.set_text(text)

    def mode_cb(self, modname, tf):
        """This method is called when a toggle button in the toolbar is pressed
        selecting one of the modes.
        """
        bm = self.canvas.get_bindmap()
        if not tf:
            bm.reset_modifier(self.canvas)
            return
        bm.set_modifier(modname, modtype='locked')
        return True

    def mode_set_cb(self, bm, modname, mtype):
        """This method is called when a mode is selected in the viewer widget.
        NOTE: it may be called when mode_cb() is not called (for example, when
        a keypress initiates a mode); however, the converse is not true: calling
        mode_cb() will always result in this method also being called as a result.

        This logic is to insure that the toggle buttons are left in a sane state
        that reflects the current mode, however it was initiated.
        """
        if modname in self.mode_actns:
            if self.mode_w and (self.mode_w != self.mode_actns[modname]):
                self.mode_w.setChecked(False)
            self.mode_w = self.mode_actns[modname]
            self.mode_w.setChecked(True)
        elif self.mode_w:
            # keystroke turned on a mode for which we have no GUI button
            # and a GUI button is selected--unselect it
            self.mode_w.setChecked(False)
            self.mode_w = None
        return True


class ColormapAction(QAction):

    def __init__(self, label, cmap, parent):
        super(ColormapAction, self).__init__(label, parent)
        self.cmap = cmap
        pm = cmap2pixmap(cmap)
        self.setIcon(QIcon(pm))


def _colormap_mode(parent, on_trigger):

    # actions for each colormap
    acts = []
    # for label, cmap in config.colormaps:
    for label in ginga_cmap.get_names():
        cmap = ginga_cmap.get_cmap(label)
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

        # self.make_toolbar()
        self.set_image(image, **kwargs)

    def _setup_axes(self):
        ## self._axes = self.central_widget.canvas.fig.add_subplot(111)
        ## self._axes.set_aspect('equal', adjustable='datalim')
        pass

    def set_image(self, image, **kwargs):
        """
        Update the image shown in the widget
        """
        if self._im is not None:
            self._im.remove()
            self._im = None

        ## kwargs.setdefault('origin', 'upper')

        # self._im = imshow(self._axes, image,
        # norm=self._norm, cmap='gray', **kwargs)
        ## self._im_array = image
        # self._axes.set_xticks([])
        # self._axes.set_yticks([])
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
        # self._im.set_cmap(cmap)
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
        # ginga takes care of this by itself
        pass


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
    tups = [cmap.clst[int(x * n)] for x in inds]
    rgbas = [QColor(int(r * 255), int(g * 255),
                    int(b * 255), 255).rgba() for r, g, b in tups]
    im = QImage(steps, 1, QImage.Format_Indexed8)
    im.setColorTable(rgbas)
    for i in range(steps):
        im.setPixel(i, 0, i)
    im = im.scaled(128, 32)
    pm = QPixmap.fromImage(im)
    return pm
