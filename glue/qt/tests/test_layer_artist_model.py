from __future__ import absolute_import, division, print_function

from ...external.qt.QtCore import Qt

from mock import MagicMock

from ..layer_artist_model import LayerArtistModel, LayerArtistView
from ...clients.layer_artist import LayerArtist as _LayerArtist
from ...core import Data


class LayerArtist(_LayerArtist):

    def update(self, view=None):
        pass


def setup_model(num):
    ax = MagicMock()
    mgrs = [LayerArtist(Data(label=str(i)), ax) for i in range(num)]
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
    mgrs = [LayerArtist(Data(label='A'), None)]
    model = LayerArtistModel(mgrs)
    model.add_artist(0, LayerArtist(Data(label='B'), None))
    assert model.rowCount() == 2


def test_add_artist_updates_artist_list():
    mgrs = [LayerArtist(Data(label='A'), None)]
    model = LayerArtistModel(mgrs)
    model.add_artist(0, LayerArtist(Data(label='B'), None))
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
    mgr = MagicMock(spec_set=LayerArtist)
    mgrs = [mgr]
    model = LayerArtistModel(mgrs)

    model.removeRow(0)
    mgr.clear.assert_called_once_with()


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
    model, _ = setup_model(1)

    assert model.flags(model.index(0)) == (Qt.ItemIsEditable |
                                           Qt.ItemIsDragEnabled |
                                           Qt.ItemIsEnabled |
                                           Qt.ItemIsSelectable |
                                           Qt.ItemIsUserCheckable)
    assert model.flags(model.index(-1)) == (Qt.ItemIsDropEnabled)


def test_move_artist_empty():
    mgrs = []
    model = LayerArtistModel(mgrs)
    model.move_artist(None, 0)

    assert mgrs == []


def test_move_artist_single():
    ax = MagicMock()
    m0 = LayerArtist(Data(label="test 0"), ax)
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
    m0 = LayerArtist(Data(label='test 0'), MagicMock())
    m1 = LayerArtist(Data(label='test 1'), MagicMock())
    m2 = LayerArtist(Data(label='test 2'), MagicMock())
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
    m0 = LayerArtist(Data(label='test 0'), MagicMock())
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
        self.view = LayerArtistView()
        self.view.setModel(self.model)

    def test_current_row(self):
        for row in [0, 1]:
            idx = self.model.index(row)
            self.view.setCurrentIndex(idx)
            self.view.current_row() == row
            assert self.view.current_artist() is self.artists[row]
