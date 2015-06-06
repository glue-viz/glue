from ....external.qt.QtCore import QPoint
from ....external.qt.QtGui import QMainWindow
from ....core import Data

from . import simple_session
from ..style_dialog import StyleDialog


class NonBlockingStyleDialog(StyleDialog):
    def exec_(self, *args):
        self.show()


def test_style_dialog():

    # This is in part a regression test for a bug in Python 3. It is not a
    # full test of StyleDialog.

    session = simple_session()
    hub = session.hub
    collect = session.data_collection

    image = Data(label='im',
                 x=[[1, 2], [3, 4]],
                 y=[[2, 3], [4, 5]])

    pos = QPoint(10, 10)
    st = NonBlockingStyleDialog.dropdown_editor(image, pos)
