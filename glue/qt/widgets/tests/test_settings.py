from ..settings_editor import SettingsEditor


class MockApplication(object):

    def __init__(self, data):
        self.data = data

    @property
    def settings(self):
        return list(self.data.items())

    def get_setting(self, key):
        return self.data[key]

    def set_setting(self, key, value):
        self.data[key] = value


class TestSettings(object):

    def setup_method(self, method):
        self.a = MockApplication({'k1': 'v1'})
        self.editor = SettingsEditor(self.a)
        self.widget = self.editor.widget

    def teardown_method(self, method):
        self.widget.close()

    def test_init(self):
        assert self.widget.item(0, 0).text() == 'k1'
        assert self.widget.item(0, 1).text() == 'v1'

    def test_set_setting(self):
        self.widget.item(0, 1).setText('v2')
        assert self.a.get_setting('k1') == 'v2'
