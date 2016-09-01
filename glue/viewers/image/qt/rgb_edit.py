from __future__ import absolute_import, division, print_function

from qtpy import QtCore, QtWidgets
from glue.core.qt.component_id_combo import ComponentIDCombo


class RGBEdit(QtWidgets.QWidget):

    """A widget to set the contrast for individual layers in an RGB image

    Based off the ds9 RGB Frame widget

    :param artist: A :class:`~glue.viewers.image.layer_artist.RGBArtistLayerArtist`
                   instance to control

    :param parent: Optional widget parent

    This widget sets the state of the artist object, such that contrast
    adjustments from a :class:`~glue.viewers.image.client` affect
    a particular RGB slice
    """
    current_changed = QtCore.Signal(str)
    colors_changed = QtCore.Signal()

    def __init__(self, parent=None, artist=None):
        super(RGBEdit, self).__init__(parent)
        self._artist = artist

        l = QtWidgets.QGridLayout()

        current = QtWidgets.QLabel("Contrast")
        visible = QtWidgets.QLabel("Visible")
        l.addWidget(current, 0, 2, 1, 1)
        l.addWidget(visible, 0, 3, 1, 1)
        l.setColumnStretch(0, 0)
        l.setColumnStretch(1, 10)
        l.setColumnStretch(2, 0)
        l.setColumnStretch(3, 0)

        l.setRowStretch(0, 0)
        l.setRowStretch(1, 0)
        l.setRowStretch(2, 0)
        l.setRowStretch(3, 0)
        l.setRowStretch(4, 10)

        curr_grp = QtWidgets.QButtonGroup()
        self.current = {}
        self.vis = {}
        self.cid = {}

        for row, color in enumerate(['red', 'green', 'blue'], 1):
            lbl = QtWidgets.QLabel(color.title())

            cid = ComponentIDCombo()

            curr = QtWidgets.QRadioButton()
            curr_grp.addButton(curr)

            vis = QtWidgets.QCheckBox()
            vis.setChecked(True)

            l.addWidget(lbl, row, 0, 1, 1)
            l.addWidget(cid, row, 1, 1, 1)
            l.addWidget(curr, row, 2, 1, 1)
            l.addWidget(vis, row, 3, 1, 1)

            curr.clicked.connect(self.update_current)
            vis.toggled.connect(self.update_visible)
            cid.currentIndexChanged.connect(self.update_layers)

            self.cid[color] = cid
            self.vis[color] = vis
            self.current[color] = curr

        self.setLayout(l)
        self.current['red'].click()

    @property
    def attributes(self):
        """A 3-tuple of the ComponentIDs for each RGB layer"""
        return tuple(self.cid[c].component for c in ['red', 'green', 'blue'])

    @attributes.setter
    def attributes(self, cids):
        for cid, c in zip(cids, ['red', 'green', 'blue']):
            if cid is None:
                continue
            self.cid[c].component = cid

    @property
    def rgb_visible(self):
        """ A 3-tuple of the visibility of each layer, as bools """
        return tuple(self.vis[c].isChecked() for c in ['red', 'green', 'blue'])

    @rgb_visible.setter
    def rgb_visible(self, value):
        for v, c in zip(value, 'red green blue'.split()):
            self.vis[c].setChecked(v)

    @property
    def artist(self):
        return self._artist

    @artist.setter
    def artist(self, value):
        self._artist = value
        for cid in self.cid.values():
            cid.data = value.layer
        self.update_layers()

    def update_layers(self):
        if self.artist is None:
            return

        r = self.cid['red'].component
        g = self.cid['green'].component
        b = self.cid['blue'].component
        changed = self.artist.r is not r or \
            self.artist.g is not g or\
            self.artist.b is not b

        self.artist.r = r
        self.artist.g = g
        self.artist.b = b

        if changed:
            self.colors_changed.emit()

        self.artist.update()
        self.artist.redraw()

    def update_current(self, *args):
        if self.artist is None:
            return

        for c in ['red', 'green', 'blue']:
            if self.current[c].isChecked():
                self.artist.contrast_layer = c
                self.current_changed.emit(c)
                break
        else:
            raise RuntimeError("Could not determine which layer is current")

    def update_visible(self, *args):
        if self.artist is None:
            return

        self.artist.layer_visible['red'] = self.vis['red'].isChecked()
        self.artist.layer_visible['green'] = self.vis['green'].isChecked()
        self.artist.layer_visible['blue'] = self.vis['blue'].isChecked()
        self.artist.update()
        self.artist.redraw()
