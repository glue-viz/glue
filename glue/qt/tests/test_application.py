from PyQt4.QtGui import QApplication

from ..glue_application import GlueApplication


def tab_count(app):
    return app.tab_bar.count()

def setup_module(module):
    module.app = QApplication([''])

def teardown_module(module):
    del module.app

class TestGlueApplication(object):

    def setup_method(self, method):
        self.app = GlueApplication()

    def teardown_method(self, method):
        self.app.close()
        del self.app

    def test_new_tabs(self):
        t0 = tab_count(self.app)
        self.app._new_tab()
        assert tab_count(self.app) == t0 + 1
