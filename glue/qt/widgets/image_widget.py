from PyQt4.QtGui import QMainWindow, QWidget, QAction, QToolBar
from PyQt4.QtCore import Qt, QVariant

import matplotlib.cm as cm

from ... import core

from ...clients.image_client import ImageClient

from ..mouse_mode import RectangleMode, CircleMode, PolyMode, \
                         ContrastMode, ContourMode
from ..glue_toolbar import GlueToolbar

from ..ui.imagewidget import Ui_ImageWidget


class ImageWidget(QMainWindow, core.hub.HubListener):

    def __init__(self, data, parent=None):

        QMainWindow.__init__(self, parent)
        core.hub.HubListener.__init__(self)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.ui = Ui_ImageWidget()
        self.ui.setupUi(self.central_widget)

        self.client = ImageClient(data,
                                  self.ui.mplWidget.canvas.fig,
                                  self.ui.mplWidget.canvas.ax)

        self._create_actions()
        self.make_toolbar()
        self._connect()
        self._init_widgets()
        self.set_data(0)
        self.set_orientation(0)
        self.statusBar().setSizeGripEnabled(False)
        self.setFocusPolicy(Qt.StrongFocus)
        self.resize(self.central_widget.size())

    def _create_actions(self):
        self.cmap_heat_action = QAction("Hot", self)
        self.cmap_heat_action.activated.connect(
            lambda: self.client.set_cmap(cm.hot))
        self.cmap_heat_action.triggered.connect(
            lambda: self.client.set_cmap(cm.hot))

        self.cmap_paired_action = QAction("Paired", self)
        self.cmap_paired_action.triggered.connect(
            lambda: self.client.set_cmap(cm.Paired))

        self.cmap_gray_action = QAction("Gray", self)
        self.cmap_gray_action.triggered.connect(
            lambda: self.client.set_cmap(cm.gray))

        self.cmap_spectral_action = QAction("Spectral", self)
        self.cmap_spectral_action.triggered.connect(
            lambda: self.client.set_cmap(cm.spectral))

        self.cmap_blue_action = QAction("Blues", self)
        self.cmap_blue_action.triggered.connect(
            lambda: self.client.set_cmap(cm.Blues))

    def create_colormap_toolbar(self, parent=None):
        self.cmap_toolbar = QToolBar("Colormaps", parent)
        self.cmap_toolbar.addAction(self.cmap_heat_action)
        self.cmap_toolbar.addAction(self.cmap_paired_action)
        self.cmap_toolbar.addAction(self.cmap_gray_action)
        self.cmap_toolbar.addAction(self.cmap_spectral_action)
        self.cmap_toolbar.addAction(self.cmap_blue_action)
        self.cmap_toolbar.addAction(self.toggle_colorbar_action)
        return self.cmap_toolbar

    def make_toolbar(self):
        result = GlueToolbar(self.ui.mplWidget.canvas, self, name='Image')
        for mode in self._mouse_modes():
            result.add_mode(mode)
        self.addToolBar(result)
        return result

    def _apply_roi(self, mode):
        roi = mode.roi()
        self.client._apply_roi(roi)

    def _mouse_modes(self):
        axes = self.ui.mplWidget.canvas.ax
        rect = RectangleMode(axes, release_callback=self._apply_roi)
        circ = CircleMode(axes, release_callback=self._apply_roi)
        poly = PolyMode(axes, release_callback=self._apply_roi)
        contrast = ContrastMode(axes, move_callback=self._set_norm)
        contour = ContourMode(axes, release_callback=self._contour_roi)
        return [rect, circ, poly, contrast, contour]

    def _init_widgets(self):
        self.ui.imageSlider.hide()
        self.ui.sliceComboBox.hide()
        self.ui.sliderLabel.hide()
        self.ui.sliceComboBox.addItems(["xy", "xz", "yz"])
        for d in self.client.data:
            self.add_data(d)

    def add_data(self, data):
        if not self.client.can_handle_data(data):
            return
        self.ui.displayDataCombo.addItem(data.label, userData=QVariant(data))

    def set_data(self, index):
        if self.ui.displayDataCombo.count() == 0:
            return

        data = self.ui.displayDataCombo.itemData(index).toPyObject()
        self.client.set_data(data)
        self.ui.displayDataCombo.setCurrentIndex(index)
        self.set_attribute_combo(data)
        if not self.client.is_3D:
            self.ui.imageSlider.hide()
            self.ui.sliderLabel.hide()
            self.ui.sliceComboBox.hide()
            self.ui.orientationLabel.hide()
        else:
            self.ui.imageSlider.show()
            self.ui.sliderLabel.show()
            self.ui.sliceComboBox.show()
            self.ui.orientationLabel.show()
        self.set_slider_range()

    def set_attribute(self, index):
        combo = self.ui.attributeComboBox
        component_id = combo.itemData(index).toPyObject()
        self.client.set_attribute(component_id)
        self.ui.attributeComboBox.setCurrentIndex(index)

    def set_attribute_combo(self, data):
        combo = self.ui.attributeComboBox
        combo.currentIndexChanged.disconnect(self.set_attribute)
        combo.clear()
        fields = data.component_ids()
        for f in fields:
            combo.addItem(f.label, userData=f)
        combo.currentIndexChanged.connect(self.set_attribute)

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
        self.client.register_to_hub(hub)
        dc_filt = lambda x: x.sender is self.client._data

        hub.subscribe(self,
                      core.message.DataCollectionAddMessage,
                      handler=lambda x: self.add_data(x.data),
                      filter=dc_filt)
        hub.subscribe(self,
                      core.message.DataCollectionDeleteMessage,
                      handler=lambda x: self.remove_data(x.data),
                      filter=dc_filt)

    def remove_data(self, data):
        combo = self.ui.displayDataCombo
        for item in range(combo.count()):
            if combo.itemData(item).toPyObject() is data:
                combo.removeItem(item)
                break

    def _set_norm(self, mode):
        """ Use the `ContrastMouseMode` to adjust the transfer function """
        im = self.client.image
        if im is None:
            return
        vlo, vhi = mode.get_scaling(im)
        return self.client.set_norm(vlo, vhi)

    def _contour_roi(self, mode):
        """ Callback for ContourMode. Set edit_subset as new ROI """
        im = self.client.image
        if im is None:
            return
        roi = mode.roi(im)
        if roi:
            self.client._apply_roi(roi)

    def __str__(self):
        return "Image Widget"
