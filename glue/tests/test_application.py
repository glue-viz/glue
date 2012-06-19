import unittest

from PyQt4.QtGui import QApplication

from glue.qt.glue_application import GlueApplication

def tab_count(app):
    return app.tab_bar.count()


class TestGlueApplication(unittest.TestCase):
    def setUp(self):
        self.qapp = QApplication([''])
        self.app = GlueApplication()

    def tearDown(self):
        self.app.close()
        del self.app
        del self.qapp

    def test_new_tabs(self):
        t0 = tab_count(self.app)
        self.app._new_tab()
        self.assertEquals(t0+1, tab_count(self.app))

if __name__ == "__main__":
    unittest.main()