# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from qtpy import QtWidgets, QtGui

from ..actions import GlueActionButton


def test_glue_action_button():
    a = QtWidgets.QAction(None)
    a.setToolTip("testtooltip")
    a.setWhatsThis("testwhatsthis")
    a.setIcon(QtGui.QIcon("dummy_file"))
    a.setText('testtext')
    b = GlueActionButton()
    b.set_action(a)

    # assert b.icon() == a.icon() icons are copied, apparently
    assert b.text() == a.text()
    assert b.toolTip() == a.toolTip()
    assert b.whatsThis() == a.whatsThis()

    # stays in sync
    a.setText('test2')
    assert b.text() == 'test2'
