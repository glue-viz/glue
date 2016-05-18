from mock import patch, MagicMock

from matplotlib.colors import ColorConverter

from glue.external.qt import QtGui
from glue.app.qt.preferences import PreferencesDialog
from glue.utils.qt.helpers import process_dialog

rgb = ColorConverter().to_rgb

class TestPreferences():


    def setup_method(self, method):
        self.app = MagicMock()

    def test_no_change(self):

        # If we don't change anything, settings should be preserved

        with patch('glue.app.qt.preferences.settings') as settings:

            settings.FOREGROUND_COLOR = 'red'
            settings.BACKGROUND_COLOR = (0, 0.5, 1)
            settings.DATA_COLOR = (1, 0.5, 0.25)
            settings.DATA_ALPHA = 0.3

            dialog = PreferencesDialog(self.app)
            dialog.show()

            assert dialog.theme == 'Custom'

            dialog.accept()

            assert rgb(settings.FOREGROUND_COLOR) == (1, 0, 0)
            assert rgb(settings.BACKGROUND_COLOR) == (0, 0.5, 1)
            assert rgb(settings.DATA_COLOR) == (1, 0.5, 0.25)
            assert settings.DATA_ALPHA == 0.3

    def test_theme_autodetect(self):

        # If we don't change anything, settings should be preserved

        with patch('glue.app.qt.preferences.settings') as settings:

            settings.FOREGROUND_COLOR = 'white'
            settings.BACKGROUND_COLOR = 'black'
            settings.DATA_COLOR = '0.75'
            settings.DATA_ALPHA = 0.8

            dialog = PreferencesDialog(self.app)
            dialog.show()
            assert dialog.theme == 'White on Black'
            dialog.accept()

            settings.FOREGROUND_COLOR = 'black'
            settings.BACKGROUND_COLOR = 'white'
            settings.DATA_COLOR = '0.35'
            settings.DATA_ALPHA = 0.8

            dialog = PreferencesDialog(self.app)
            dialog.show()
            assert dialog.theme == 'Black on White'
            dialog.accept()

    def test_themes(self):

        # Check that themes work

        with patch('glue.app.qt.preferences.settings') as settings:

            settings.FOREGROUND_COLOR = 'red'
            settings.BACKGROUND_COLOR = (0, 0.5, 1)
            settings.DATA_COLOR = (1, 0.5, 0.25)
            settings.DATA_ALPHA = 0.3

            dialog = PreferencesDialog(self.app)
            dialog.show()
            dialog.theme = 'White on Black'
            dialog.accept()

            assert rgb(settings.FOREGROUND_COLOR) == (1, 1, 1)
            assert rgb(settings.BACKGROUND_COLOR) == (0, 0, 0)
            assert rgb(settings.DATA_COLOR) == (0.75, 0.75, 0.75)
            assert settings.DATA_ALPHA == 0.8

            dialog = PreferencesDialog(self.app)
            dialog.show()
            dialog.theme = 'Black on White'
            dialog.accept()

            assert rgb(settings.FOREGROUND_COLOR) == (0, 0, 0)
            assert rgb(settings.BACKGROUND_COLOR) == (1, 1, 1)
            assert rgb(settings.DATA_COLOR) == (0.35, 0.35, 0.35)
            assert settings.DATA_ALPHA == 0.8

    def test_custom_changes(self):

        # Check that themes work

        with patch('glue.app.qt.preferences.settings') as settings:

            settings.FOREGROUND_COLOR = 'red'
            settings.BACKGROUND_COLOR = (0, 0.5, 1)
            settings.DATA_COLOR = (1, 0.5, 0.25)
            settings.DATA_ALPHA = 0.3

            dialog = PreferencesDialog(self.app)
            dialog.show()
            dialog.foreground = (0,1,1)
            dialog.accept()

            assert rgb(settings.FOREGROUND_COLOR) == (0, 1, 1)
            assert rgb(settings.BACKGROUND_COLOR) == (0, 0.5, 1)
            assert rgb(settings.DATA_COLOR) == (1, 0.5, 0.25)
            assert settings.DATA_ALPHA == 0.3

            dialog = PreferencesDialog(self.app)
            dialog.show()
            dialog.background = (1,0,1)
            dialog.accept()

            assert rgb(settings.FOREGROUND_COLOR) == (0, 1, 1)
            assert rgb(settings.BACKGROUND_COLOR) == (1, 0, 1)
            assert rgb(settings.DATA_COLOR) == (1, 0.5, 0.25)
            assert settings.DATA_ALPHA == 0.3

            dialog = PreferencesDialog(self.app)
            dialog.show()
            dialog.data_color = (1,1,0.5)
            dialog.accept()

            assert rgb(settings.FOREGROUND_COLOR) == (0, 1, 1)
            assert rgb(settings.BACKGROUND_COLOR) == (1, 0, 1)
            assert rgb(settings.DATA_COLOR) == (1, 1, 0.5)
            assert settings.DATA_ALPHA == 0.3

            dialog = PreferencesDialog(self.app)
            dialog.show()
            dialog.data_alpha = 0.4
            dialog.accept()

            assert rgb(settings.FOREGROUND_COLOR) == (0, 1, 1)
            assert rgb(settings.BACKGROUND_COLOR) == (1, 0, 1)
            assert rgb(settings.DATA_COLOR) == (1, 1, 0.5)
            assert settings.DATA_ALPHA == 0.4

    def test_custom_pane(self):

        settings = MagicMock()

        class CustomPreferences(QtGui.QWidget):

            def __init__(self, parent=None):

                super(CustomPreferences, self).__init__(parent=parent)

                self.layout = QtGui.QFormLayout()

                self.option1 = QtGui.QLineEdit()
                self.option2 = QtGui.QLineEdit()

                self.layout.addRow("Option 1", self.option1)
                self.layout.addRow("Option 2", self.option2)

                self.setLayout(self.layout)

            def finalize(self):
                settings.OPTION1 = "Monty"
                settings.OPTION2 = "Python"

        preference_panes = [('Custom', CustomPreferences)]

        with patch('glue.app.qt.preferences.preference_panes', preference_panes):

            dialog = PreferencesDialog(self.app)
            dialog.show()
            dialog.accept()

            assert settings.OPTION1 == "Monty"
            assert settings.OPTION2 == "Python"
