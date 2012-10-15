from PyQt4.QtGui import QListView
from PyQt4.QtCore import Qt

from mock import MagicMock

from ..layer_artist_model import LayerArtistModel
from ...clients.layer_artist import LayerArtist
from ...core import Data


def test_row_count():
    d = Data()
    mgrs = []
    assert LayerArtistModel(mgrs).rowCount() == 0
    mgrs.append(LayerArtist(d, None))
    assert LayerArtistModel(mgrs).rowCount() == 1
    mgrs.append(LayerArtist(d, None))
    assert LayerArtistModel(mgrs).rowCount() == 2


def test_row_label():
    mgrs = []
    for i in range(5):
        d = Data(label="Test %i" % i)
        mgrs.append(LayerArtist(d, None))

    model = LayerArtistModel(mgrs)
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
    mgr = LayerArtist(Data(label='test 0'), None)
    mgrs = [mgr]
    model = LayerArtistModel(mgrs)

    lbl = mgr.layer.label
    model.change_label(0, 'new label')
    assert mgr.layer.label != lbl


def test_change_label_invalid_row():
    mgr = LayerArtist(Data(label='test 0'), None)
    mgrs = [mgr]
    model = LayerArtistModel(mgrs)

    lbl = mgr.layer.label
    model.change_label(1, 'new label')
    assert mgr.layer.label == lbl


def test_flags():
    mgr = LayerArtist(Data(label='test 0'), None)
    mgrs = [mgr]
    model = LayerArtistModel(mgrs)

    assert model.flags(model.index(0)) == (Qt.ItemIsEditable |
                                           Qt.ItemIsDragEnabled |
                                           Qt.ItemIsDropEnabled |
                                           Qt.ItemIsEnabled |
                                           Qt.ItemIsSelectable |
                                           Qt.ItemIsUserCheckable)


def test_move_artist():
    ax = MagicMock()
    m0 = LayerArtist(Data(label='test 0'), ax)
    m1 = LayerArtist(Data(label='test 1'), ax)
    m2 = LayerArtist(Data(label='test 2'), ax)
    mgrs = [m0, m1, m2]
    model = LayerArtistModel(mgrs)

    model.move_artist(m0, 1)
    print mgrs
    assert mgrs == [m1, m0, m2]

    model.move_artist(m0, 1)
    print mgrs
    assert mgrs == [m1, m0, m2]

    model.move_artist(m0, 0)
    print mgrs
    assert mgrs == [m0, m1, m2]

    model.move_artist(m2, 0)
    print mgrs
    assert mgrs == [m2, m0, m1]

    model.move_artist(m1, 1)
    print mgrs
    assert mgrs == [m2, m1, m0]


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
