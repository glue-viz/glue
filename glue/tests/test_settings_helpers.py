from mock import patch

import os

from glue.config import SettingRegistry
from glue._settings_helpers import load_settings, save_settings


def test_roundtrip(tmpdir):

    settings = SettingRegistry()

    settings.add('STRING', 'green', str)
    settings.add('INT', 3, int)
    settings.add('FLOAT', 5.5, float)
    settings.add('LIST', [1,2,3], list)

    with patch('glue.config.settings', settings):
        with patch('glue.config.CFG_DIR', tmpdir.strpath):

            settings.STRING = 'blue'
            settings.INT = 4
            settings.FLOAT = 3.5
            settings.LIST = ['A', 'BB', 'CCC']

            settings.reset_defaults()

            assert settings.STRING == 'green'
            assert settings.INT == 3
            assert settings.FLOAT == 5.5
            assert settings.LIST == [1, 2, 3]

            settings.STRING = 'blue'
            settings.INT = 4
            settings.FLOAT = 3.5
            settings.LIST = ['A', 'BB', 'CCC']

            save_settings()

            assert os.path.exists(os.path.join(tmpdir.strpath, 'settings.cfg'))

            settings.reset_defaults()

            settings.STRING = 'red'
            settings.INT = 5

            # Loading settings will only change settings that have not been 
            # changed from the defaults...
            load_settings()

            assert settings.STRING == 'red'
            assert settings.INT == 5
            assert settings.FLOAT == 3.5
            assert settings.LIST == ['A', 'BB', 'CCC']

            # ... unless the ``force=True`` option is passed
            load_settings(force=True)

            assert settings.STRING == 'blue'
            assert settings.INT == 4
            assert settings.FLOAT == 3.5
            assert settings.LIST == ['A', 'BB', 'CCC']
