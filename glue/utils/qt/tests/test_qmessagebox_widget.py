from .. import QMessageBoxPatched as QMessageBox
from glue.qt import get_qapp


def test_main():

    app = get_qapp()

    w = QMessageBox(QMessageBox.Critical, "Error", "An error occurred")
    w.setDetailedText("Spam")
    w.select_all()
    w.copy_detailed()

    assert app.clipboard().text() == "Spam"

    app.quit()
