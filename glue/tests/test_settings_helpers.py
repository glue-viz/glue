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

            save_settings()

            assert os.path.exists(os.path.join(tmpdir.strpath, 'settings.cfg'))

            settings.STRING = 'red'
            settings.INT = 3
            settings.FLOAT = 4.5
            settings.LIST = ['DDD', 'EE', 'F']

            load_settings(force=True)

            assert settings.STRING == 'blue'
            assert settings.INT == 4
            assert settings.FLOAT == 3.5
            assert settings.LIST == ['A', 'BB', 'CCC']
