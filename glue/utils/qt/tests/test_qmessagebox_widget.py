from .. import QMessageBoxPatched as QMessageBox
from ....qt import get_qapp
from ....external.qt import QtGui


def test_main():

    app = get_qapp()

    w = QMessageBox(QMessageBox.Critical, "Error", "An error occurred")
    w.setDetailedText("Spam")
    w.select_all()
    w.copy_detailed()

    assert app.clipboard().text() == "Spam"

    app.quit()
