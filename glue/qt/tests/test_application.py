from PyQt4.QtGui import QApplication

from ..glue_application import GlueApplication

def tab_count(app):
    return app.tab_bar.count()


class TestGlueApplication(object):

    def setup_method(self, method):
        self.qapp = QApplication([''])
        self.app = GlueApplication()

    def tearDown(self):
        self.app.close()
        del self.app
        del self.qapp

    def test_new_tabs(self):
        t0 = tab_count(self.app)
        self.app._new_tab()
        assert tab_count(self.app) == t0+1
