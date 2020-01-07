from unittest.mock import MagicMock

from qtpy.QtCore import Qt
from glue.core import Data, Hub
from glue.core.layer_artist import LayerArtistBase as _LayerArtist

from ..layer_artist_model import LayerArtistModel, LayerArtistView


class LayerArtist(_LayerArtist):

    update_count = 0
    clear_count = 0
    redraw_count = 0

    def update(self):
        self.update_count += 1

    def clear(self):
        self.clear_count += 1

    def redraw(self):
        self.redraw_count += 1


def setup_model(num):
    mgrs = [LayerArtist(Data(label=str(i))) for i in range(num)]
    model = LayerArtistModel(mgrs)
    return model, mgrs


def test_row_count():
    for n in range(4):
        assert setup_model(n)[0].rowCount() == n


def test_row_label():
    model, mgrs = setup_model(5)
    for i in range(5):
        assert model.row_label(i) == mgrs[i].layer.label


def test_add_artist_updates_row_count():
    mgrs = [LayerArtist(Data(label='A'))]
    model = LayerArtistModel(mgrs)
    model.add_artist(0, LayerArtist(Data(label='B')))
    assert model.rowCount() == 2


def test_add_artist_updates_artist_list():
    mgrs = [LayerArtist(Data(label='A'))]
    model = LayerArtistModel(mgrs)
    model.add_artist(0, LayerArtist(Data(label='B')))
    assert len(mgrs) == 2


def test_valid_remove():
    mgr = MagicMock(spec_set=LayerArtist)
    mgrs = [mgr]
    model = LayerArtistModel(mgrs)
    remove = model.removeRow(0)
    assert remove
    assert mgr not in mgrs


def test_invalid_remove():
    mgr = MagicMock(spec_set=LayerArtist)
    mgrs = [mgr]
    model = LayerArtistModel(mgrs)
    remove = model.removeRow(1)
    assert not remove
    assert mgr in mgrs


def test_artist_cleared_on_remove():
    mgr = LayerArtist(None)
    mgrs = [mgr]
    model = LayerArtistModel(mgrs)
    model.removeRow(0)
    assert mgr.clear_count == 1


def test_change_label():
    model, (mgr,) = setup_model(1)
    lbl = mgr.layer.label
    model.change_label(0, 'new label')
    assert mgr.layer.label != lbl


def test_change_label_invalid_row():
    model, (mgr,) = setup_model(1)
    lbl = mgr.layer.label
    model.change_label(1, 'new label')
    assert mgr.layer.label == lbl


def test_flags():
    model, layer_artists = setup_model(1)

    expected = (Qt.ItemIsEditable |
                Qt.ItemIsDragEnabled |
                Qt.ItemIsEnabled |
                Qt.ItemIsSelectable |
                Qt.ItemIsUserCheckable |
                Qt.ItemNeverHasChildren)

    assert model.flags(model.index(0)) == expected


def test_move_artist_empty():
    mgrs = []
    model = LayerArtistModel(mgrs)
    model.move_artist(None, 0)

    assert mgrs == []


def test_move_artist_single():
    m0 = LayerArtist(Data(label="test 0"))
    mgrs = [m0]

    model = LayerArtistModel(mgrs)
    model.move_artist(m0, 0)
    assert mgrs == [m0]
    model.move_artist(m0, -1)
    assert mgrs == [m0]
    model.move_artist(m0, 1)
    assert mgrs == [m0]
    model.move_artist(m0, 2)
    assert mgrs == [m0]


def test_move_artist_two():
    model, mgrs = setup_model(2)
    m0, m1 = mgrs

    model.move_artist(m0, 0)
    assert mgrs == [m0, m1]

    model.move_artist(m0, 1)
    assert mgrs == [m0, m1]

    model.move_artist(m0, 2)
    assert mgrs == [m1, m0]

    model.move_artist(m0, 0)
    assert mgrs == [m0, m1]


def test_move_artist_three():
    model, mgrs = setup_model(3)
    m0, m1, m2 = mgrs

    model.move_artist(m0, 0)
    assert mgrs == [m0, m1, m2]

    model.move_artist(m0, 1)
    assert mgrs == [m0, m1, m2]

    model.move_artist(m0, 2)
    assert mgrs == [m1, m0, m2]
    model.move_artist(m0, 0)

    model.move_artist(m0, 3)
    assert mgrs == [m1, m2, m0]
    model.move_artist(m0, 0)

    model.move_artist(m2, 0)
    assert mgrs == [m2, m0, m1]


def test_move_updates_zorder():
    m0 = LayerArtist(Data(label='test 0'))
    m1 = LayerArtist(Data(label='test 1'))
    m2 = LayerArtist(Data(label='test 2'))
    m0.zorder = 10
    m1.zorder = 20
    m2.zorder = 30

    mgrs = [m0, m1, m2]
    model = LayerArtistModel(mgrs)

    model.move_artist(m2, 0)
    assert m2.zorder == 30
    assert m0.zorder == 20
    assert m1.zorder == 10


def test_check_syncs_to_visible():
    m0 = LayerArtist(Data(label='test 0'))
    m0.artists = [MagicMock()]
    mgrs = [m0]
    model = LayerArtistModel(mgrs)

    m0.visible = True
    assert m0.visible
    assert model.data(model.index(0), Qt.CheckStateRole) == Qt.Checked
    m0.visible = False
    assert not m0.visible
    assert model.data(model.index(0), Qt.CheckStateRole) == Qt.Unchecked

    model.setData(model.index(0), Qt.Checked, Qt.CheckStateRole)
    assert m0.visible


def test_data():

    model, mgrs = setup_model(3)
    idx = model.index(3)
    assert model.data(idx, Qt.DisplayRole) is None
    idx = model.index(1)
    assert model.data(idx, Qt.DisplayRole) == model.row_label(1)
    assert model.data(idx, Qt.EditRole) == model.row_label(1)


class TestLayerArtistView(object):

    def setup_method(self, method):
        self.model, self.artists = setup_model(2)
        self.hub = Hub()
        self.view = LayerArtistView(hub=self.hub)
        self.view.setModel(self.model)

    def test_current_row(self):
        for row in [0, 1]:
            idx = self.model.index(row)
            self.view.setCurrentIndex(idx)
            self.view.current_row() == row
            assert self.view.current_artist() is self.artists[row]
