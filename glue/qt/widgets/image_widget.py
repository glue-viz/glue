from __future__ import absolute_import, division, print_function

from ...external.qt.QtGui import (QAction, QLabel, QCursor, QMainWindow,
                                  QToolButton, QIcon, QMessageBox
                                  )

from ...external.qt.QtCore import Qt, QRect, Signal

from .data_viewer import DataViewer
from ... import core

from ...clients.image_client import MplImageClient
from ...clients.ds9norm import DS9Normalize
from ...external.modest_image import imshow

from ...clients.layer_artist import Pointer, ImageLayerArtist
from ...core.callback_property import add_callback, delay_callback

from .data_slice_widget import DataSlice

from ..mouse_mode import (RectangleMode, CircleMode, PolyMode,
                          ContrastMode)
from ..glue_toolbar import GlueToolbar
from .mpl_widget import MplWidget, defer_draw

from ..qtutil import cmap2pixmap, load_ui, get_icon, nonpartial, update_combobox
from ..widget_properties import CurrentComboProperty, ButtonProperty, connect_current_combo
from .glue_mdi_area import GlueMdiSubWindow

WARN_THRESH = 10000000  # warn when contouring large images

__all__ = ['ImageWidget']


class ImageWidgetBase(DataViewer):

    """
    Widget for ImageClient

    This base class avoids any matplotlib-specific logic
    """

    LABEL = "Image Viewer"
    _property_set = DataViewer._property_set + \
        'data attribute rgb_mode rgb_viz ratt gatt batt slice'.split()

    attribute = CurrentComboProperty('ui.attributeComboBox',
                                     'Current attribute')
    data = CurrentComboProperty('ui.displayDataCombo',
                                'Current data')
    aspect_ratio = CurrentComboProperty('ui.aspectCombo',
                                        'Aspect ratio for image')
    rgb_mode = ButtonProperty('ui.rgb',
                              'RGB Mode?')
    rgb_viz = Pointer('ui.rgb_options.rgb_visible')

    def __init__(self, session, parent=None):
        super(ImageWidgetBase, self).__init__(session, parent)
        self._setup_widgets()
        self.client = self.make_client()

        self._setup_tools()

        tb = self.make_toolbar()
        self.addToolBar(tb)

        self._connect()

    def _setup_widgets(self):
        self.central_widget = self.make_central_widget()
        self.label_widget = QLabel("", self.central_widget)
        self.setCentralWidget(self.central_widget)
        self.ui = load_ui('imagewidget', None)
        self.option_widget = self.ui
        self.ui.slice = DataSlice()
        self.ui.slice_layout.addWidget(self.ui.slice)
        self._tweak_geometry()

        self.ui.aspectCombo.addItem("Square Pixels", userData='equal')
        self.ui.aspectCombo.addItem("Automatic", userData='auto')

    def make_client(self):
        """ Instantiate and return an ImageClient subclass """
        raise NotImplementedError()

    def make_central_widget(self):
        """ Create and return the central widget to display the image """
        raise NotImplementedError()

    def make_toolbar(self):
        """ Create and return the toolbar for this widget """
        raise NotImplementedError()

    @staticmethod
    def _get_default_tools():
        return []

    def _setup_tools(self):
        """
        Set up additional tools for this widget
        """
        from ... import config
        self._tools = []
        for tool in config.tool_registry.members[self.__class__]:
            self._tools.append(tool(self))

    def _tweak_geometry(self):
        self.central_widget.resize(600, 400)
        self.resize(self.central_widget.size())
        self.ui.rgb_options.hide()
        self.statusBar().setSizeGripEnabled(False)
        self.setFocusPolicy(Qt.StrongFocus)

    @defer_draw
    def add_data(self, data):
        """
        Add a new dataset to the viewer
        """
        # overloaded from DataViewer

        # need to delay callbacks, otherwise might
        # try to set combo boxes to nonexisting items
        with delay_callback(self.client, 'display_data', 'display_attribute'):

            # If there is not already any image data set, we can't add 1-D
            # datasets (tables/catalogs) to the image widget yet.
            if data.data.ndim == 1 and self.client.display_data is None:
                QMessageBox.information(self.window(), "Note",
                                        "Cannot create image viewer from a 1-D "
                                        "dataset. You will need to first "
                                        "create an image viewer using data "
                                        "with 2 or more dimensions, after "
                                        "which you will be able to overlay 1-D "
                                        "data as a scatter plot.",
                                        buttons=QMessageBox.Ok)
                return

            r = self.client.add_layer(data)
            if r is not None and self.client.display_data is not None:
                self.add_data_to_combo(data)
                if self.client.can_image_data(data):
                    self.client.display_data = data
                self.set_attribute_combo(self.client.display_data)

        return r is not None

    @defer_draw
    def add_subset(self, subset):
        self.client.add_scatter_layer(subset)
        assert subset in self.client.artists

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

    @property
    def slice(self):
        return self.client.slice

    @slice.setter
    def slice(self, value):
        self.client.slice = value

    def set_attribute_combo(self, data):
        """ Update attribute combo box to reflect components in data"""
        labeldata = ((f.label, f) for f in data.visible_components)
        update_combobox(self.ui.attributeComboBox, labeldata)

    def _connect(self):
        ui = self.ui

        ui.monochrome.toggled.connect(self._update_rgb_console)
        ui.rgb_options.colors_changed.connect(self.update_window_title)

        # sync client and widget slices
        ui.slice.slice_changed.connect(lambda: setattr(self, 'slice', self.ui.slice.slice))
        update_ui_slice = lambda val: setattr(ui.slice, 'slice', val)
        add_callback(self.client, 'slice', update_ui_slice)
        add_callback(self.client, 'display_data', self.ui.slice.set_data)

        # sync window title to data/attribute
        add_callback(self.client, 'display_data', nonpartial(self._display_data_changed))
        add_callback(self.client, 'display_attribute', nonpartial(self._display_attribute_changed))
        add_callback(self.client, 'display_aspect', nonpartial(self.client._update_aspect))

        # sync data/attribute combos with client properties
        connect_current_combo(self.client, 'display_data', self.ui.displayDataCombo)
        connect_current_combo(self.client, 'display_attribute', self.ui.attributeComboBox)
        connect_current_combo(self.client, 'display_aspect', self.ui.aspectCombo)

    def _display_data_changed(self, *args):

        if self.client.display_data is None:
            self.ui.attributeComboBox.clear()
            return

        with self.client.artists.ignore_empty():
            self.set_attribute_combo(self.client.display_data)
        self.client.add_layer(self.client.display_data)
        self.client._update_and_redraw()
        self.update_window_title()

    def _display_attribute_changed(self, *args):

        if self.client.display_attribute is None:
            return

        self.client._update_and_redraw()
        self.update_window_title()

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
        super(ImageWidgetBase, self).register_to_hub(hub)
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
        super(ImageWidgetBase, self).unregister(hub)
        for obj in [self, self.client]:
            hub.unsubscribe_all(obj)

    def remove_data_from_combo(self, data):
        """ Remove a data object from the combo box, if present """
        combo = self.ui.displayDataCombo
        pos = combo.findText(data.label)
        if pos >= 0:
            combo.removeItem(pos)

    def _set_norm(self, mode):
        """ Use the `ContrastMouseMode` to adjust the transfer function """

        # at least one of the clip/vmin pairs will be None
        clip_lo, clip_hi = mode.get_clip_percentile()
        vmin, vmax = mode.get_vmin_vmax()
        stretch = mode.stretch
        return self.client.set_norm(clip_lo=clip_lo, clip_hi=clip_hi,
                                    stretch=stretch,
                                    vmin=vmin, vmax=vmax,
                                    bias=mode.bias, contrast=mode.contrast)

    @property
    def window_title(self):
        if self.client.display_data is None or self.client.display_attribute is None:
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

        return title

    def _sync_data_combo_labels(self):
        combo = self.ui.displayDataCombo
        for i in range(combo.count()):
            combo.setItemText(i, combo.itemData(i).label)

    def _sync_data_labels(self):
        self.update_window_title()
        self._sync_data_combo_labels()

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
        with delay_callback(self.client, 'display_data', 'display_attribute'):
            self.client.restore_layers(rec, context)

            for artist in self.layers:
                self.add_data_to_combo(artist.layer.data)

            self.set_attribute_combo(self.client.display_data)

        self._sync_data_combo_labels()

    def closeEvent(self, event):
        # close window and all plugins
        super(ImageWidgetBase, self).closeEvent(event)
        if event.isAccepted():
            for t in self._tools:
                t.close()


class ImageWidget(ImageWidgetBase):

    """
    A matplotlib-based image widget
    """

    def make_client(self):
        return MplImageClient(self._data,
                              self.central_widget.canvas.fig,
                              artist_container=self._container)

    def make_central_widget(self):
        return MplWidget()

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

        return result

    def _mouse_modes(self):

        axes = self.client.axes

        def apply_mode(mode):
            for roi_mode in roi_modes:
                if roi_mode != mode:
                    roi_mode._roi_tool.reset()
            self.apply_roi(mode.roi())

        rect = RectangleMode(axes, roi_callback=apply_mode)
        circ = CircleMode(axes, roi_callback=apply_mode)
        poly = PolyMode(axes, roi_callback=apply_mode)
        roi_modes = [rect, circ, poly]

        contrast = ContrastMode(axes, move_callback=self._set_norm)

        self._contrast = contrast

        # Get modes from tools
        tool_modes = []
        for tool in self._tools:
            tool_modes += tool._get_modes(axes)
            add_callback(self.client, 'display_data', tool._display_data_hook)

        return [rect, circ, poly, contrast] + tool_modes

    def paintEvent(self, event):
        super(ImageWidget, self).paintEvent(event)
        pos = self.central_widget.canvas.mapFromGlobal(QCursor.pos())
        x, y = pos.x(), self.central_widget.canvas.height() - pos.y()
        self._update_intensity_label(x, y)

    def _intensity_label(self, x, y):
        x, y = self.client.axes.transData.inverted().transform([(x, y)])[0]
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

    def _connect(self):
        super(ImageWidget, self)._connect()
        self.ui.rgb_options.current_changed.connect(lambda: self._toolbars[0].set_mode(self._contrast))
        self.central_widget.canvas.resize_end.connect(self.client.check_update)


class ColormapAction(QAction):

    def __init__(self, label, cmap, parent):
        super(ColormapAction, self).__init__(label, parent)
        self.cmap = cmap
        pm = cmap2pixmap(cmap)
        self.setIcon(QIcon(pm))


def _colormap_mode(parent, on_trigger):

    from ... import config

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

    def __init__(self, image=None, wcs=None, parent=None, **kwargs):
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

        if image is not None:
            self.set_image(image=image, wcs=wcs, **kwargs)

    def _setup_axes(self):
        from ...clients.viz_client import init_mpl
        _, self._axes = init_mpl(self.central_widget.canvas.fig, axes=None, wcs=True)
        self._axes.set_aspect('equal', adjustable='datalim')

    def set_image(self, image=None, wcs=None, **kwargs):
        """
        Update the image shown in the widget
        """
        if self._im is not None:
            self._im.remove()
            self._im = None

        kwargs.setdefault('origin', 'upper')

        if wcs is not None:
            self._axes.reset_wcs(wcs)
        self._im = imshow(self._axes, image, norm=self._norm, cmap='gray', **kwargs)
        self._im_array = image
        self._wcs = wcs
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
        Embed this widget in a GlueMdiSubWindow
        """
        sub = GlueMdiSubWindow()
        sub.setWidget(self)
        self.destroyed.connect(sub.close)
        self.window_closed.connect(sub.close)
        sub.resize(self.size())
        self._mdi_wrapper = sub

        return sub

    def closeEvent(self, event):
        self.window_closed.emit()
        return super(StandaloneImageWidget, self).closeEvent(event)

    def _set_norm(self, mode):
        """ Use the `ContrastMouseMode` to adjust the transfer function """
        clip_lo, clip_hi = mode.get_clip_percentile()
        vmin, vmax = mode.get_vmin_vmax()
        stretch = mode.stretch
        self._norm.clip_lo = clip_lo
        self._norm.clip_hi = clip_hi
        self._norm.stretch = stretch
        self._norm.bias = mode.bias
        self._norm.contrast = mode.contrast
        self._norm.vmin = vmin
        self._norm.vmax = vmax
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
