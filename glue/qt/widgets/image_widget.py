from functools import partial

from ...external.qt.QtGui import (QWidget, QAction,
                                  QToolButton, QIcon, QMessageBox)
from ...external.qt.QtCore import Qt

import matplotlib.cm as cm

from .data_viewer import DataViewer
from ... import core
from ... import config

from ...clients.image_client import ImageClient

from ..mouse_mode import (RectangleMode, CircleMode, PolyMode,
                          ContrastMode, ContourMode)
from ..glue_toolbar import GlueToolbar
from .mpl_widget import MplWidget


from ..decorators import set_cursor
from ..qtutil import cmap2pixmap, select_rgb, load_ui, get_icon

WARN_THRESH = 10000000  # warn when contouring large images


class ImageWidget(DataViewer):
    LABEL = "Image Viewer"

    def __init__(self, data, parent=None):
        super(ImageWidget, self).__init__(data, parent)

        self.central_widget = MplWidget()
        self.option_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.ui = load_ui('imagewidget', self.option_widget)
        self.client = ImageClient(data,
                                  self.central_widget.canvas.fig,
                                  artist_container=self._container)
        self._tweak_geometry()

        self._create_actions()
        self.make_toolbar()
        self._connect()
        self._init_widgets()
        self.set_data(0)
        self.set_orientation(0)
        self.statusBar().setSizeGripEnabled(False)
        self.setFocusPolicy(Qt.StrongFocus)

    def _tweak_geometry(self):
        self.central_widget.resize(600, 400)
        self.resize(self.central_widget.size())

    def _create_actions(self):
        #pylint: disable=E1101
        def act(name, cmap):
            a = QAction(name, self)
            a.triggered.connect(lambda *args: self.client.set_cmap(cmap))
            pm = cmap2pixmap(cmap)
            a.setIcon(QIcon(pm))
            return a

        self._cmaps = []
        for label, cmap in config.colormaps:
            self._cmaps.append(act(label,cmap))
        self._rgb_add = QAction('RGB', self)
        self._rgb_add.triggered.connect(self._add_rgb)

    def _add_rgb(self):
        drgb = select_rgb(self._data, default=self.current_data)
        if drgb is not None:
            self.client.add_rgb_layer(*drgb)

    def make_toolbar(self):
        result = GlueToolbar(self.central_widget.canvas, self, name='Image')
        for mode in self._mouse_modes():
            result.add_mode(mode)

        tb = QToolButton()
        tb.setWhatsThis("Set color scale")
        tb.setToolTip("Set color scale")
        icon = get_icon('glue_rainbow')
        tb.setIcon(icon)
        tb.setPopupMode(QToolButton.InstantPopup)
        tb.addActions(self._cmaps)
        result.addWidget(tb)

        result.addAction(self._rgb_add)

        #connect viewport update buttons to client commands to
        #allow resampling
        cl = self.client
        result.buttons['HOME'].triggered.connect(cl.check_update)
        result.buttons['FORWARD'].triggered.connect(cl.check_update)
        result.buttons['BACK'].triggered.connect(cl.check_update)

        self.addToolBar(result)
        return result

    @set_cursor(Qt.WaitCursor)
    def apply_roi(self, mode):
        roi = mode.roi()
        self.client.apply_roi(roi)

    def _mouse_modes(self):
        axes = self.client.axes
        rect = RectangleMode(axes, roi_callback=self.apply_roi)
        circ = CircleMode(axes, roi_callback=self.apply_roi)
        poly = PolyMode(axes, roi_callback=self.apply_roi)
        contrast = ContrastMode(axes, move_callback=self._set_norm)
        contour = ContourMode(axes, release_callback=self._contour_roi)
        return [rect, circ, poly, contour, contrast]

    def _init_widgets(self):
        self.ui.imageSlider.hide()
        self.ui.sliceComboBox.hide()
        self.ui.sliceComboBox.addItems(["xy", "xz", "yz"])

    def add_data(self, data):
        """Private method to ingest new data into widget"""
        self.client.add_layer(data)
        self.add_data_to_combo(data)
        self.set_data(self._data_index(data))
        return True

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
    def current_data(self):
        if self.ui.displayDataCombo.count() == 0:
            return

        index = self.ui.displayDataCombo.currentIndex()
        return self.ui.displayDataCombo.itemData(index)

    def set_data(self, index):
        if index is None:
            return

        if self.ui.displayDataCombo.count() == 0:
            return

        data = self.ui.displayDataCombo.itemData(index)
        self.client.set_data(data)
        self.ui.displayDataCombo.setCurrentIndex(index)
        self.set_attribute_combo(data)
        if not self.client.is_3D:
            self.ui.imageSlider.hide()
            self.ui.sliceComboBox.hide()
            self.ui.orientationLabel.hide()
        else:
            self.ui.imageSlider.show()
            self.ui.sliceComboBox.show()
            self.ui.orientationLabel.show()
        self.set_slider_range()
        self._update_window_title()

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

    def set_slider(self, index):
        self.client.slice_ind = index
        self.ui.imageSlider.setValue(index)

    def set_orientation(self, ori):
        # ignore for 2D data (sometimes gets triggered when widgets
        # switch state)
        if not self.client.is_3D:
            return
        self.client.set_slice_ori(ori)
        self.ui.sliceComboBox.setCurrentIndex(ori)
        self.set_slider_range()

    def set_slider_range(self):
        self.ui.imageSlider.setRange(*self.client.slice_bounds())

    def _connect(self):
        ui = self.ui

        ui.displayDataCombo.currentIndexChanged.connect(self.set_data)
        ui.attributeComboBox.currentIndexChanged.connect(self.set_attribute)
        ui.sliceComboBox.currentIndexChanged.connect(self.set_orientation)
        ui.imageSlider.sliderMoved.connect(self.set_slider)

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

    @set_cursor(Qt.WaitCursor)
    def _contour_roi(self, mode):
        """ Callback for ContourMode. Set edit_subset as new ROI """
        im = self.client.display_data
        att = self.client.display_attribute

        if im is None or att is None:
            return
        if im.size > WARN_THRESH and not self._confirm_large_image(im):
            return

        roi = mode.roi(im[att])
        if roi:
            self.client.apply_roi(roi)

    def _update_window_title(self):
        if self.client.display_data is None:
            title = ''
        else:
            title = "%s - %s" % (self.client.display_data.label,
                                 self.client.display_attribute.label)
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
        """Ask user to confirm expensive contour operations

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
