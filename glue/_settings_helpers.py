from __future__ import absolute_import, division, print_function

import os
import json

from glue.external.six.moves import configparser
from glue.logger import logger


def save_settings():

    from glue.config import settings, CFG_DIR
    settings_cfg = os.path.join(CFG_DIR, 'settings.cfg')

    config = configparser.ConfigParser()
    config.add_section('settings')

    for name, value, _ in sorted(settings):
        config.set('settings', name, value=json.dumps(value))

    if not os.path.exists(CFG_DIR):
        os.mkdir(CFG_DIR)

    with open(settings_cfg, 'w') as fout:
        config.write(fout)


def load_settings():

    from glue.config import settings, CFG_DIR
    settings_cfg = os.path.join(CFG_DIR, 'settings.cfg')

    config = configparser.ConfigParser()
    read = config.read(settings_cfg)

    if len(read) == 0 or not config.has_section('settings'):
        return

    for name, value in config.items('settings'):
        name = name.upper()
        if hasattr(settings, name):
            setattr(settings, name, json.loads(value))
        else:
            logger.info("Unknown setting {0} - skipping".format(name))
