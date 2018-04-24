from __future__ import absolute_import, division, print_function

import os
import json

from glue.external.six.moves import configparser
from glue.logger import logger


def save_settings():

    from glue.config import settings, CFG_DIR

    if not getattr(settings, '_save_to_disk', True):
        return

    settings_cfg = os.path.join(CFG_DIR, 'settings.cfg')

    config = configparser.ConfigParser()
    config.add_section('main')

    for name, value, _ in sorted(settings):
        config.set('main', name, value=json.dumps(value, sort_keys=True))

    if not os.path.exists(CFG_DIR):
        os.mkdir(CFG_DIR)

    with open(settings_cfg, 'w') as fout:
        config.write(fout)


def load_settings(force=False):
    """
    Load the settings from disk.

    By default, only settings not already defined in memory are read in, but
    by setting ``force=True``, all settings will be read in.
    """

    from glue.config import settings, CFG_DIR
    settings_cfg = os.path.join(CFG_DIR, 'settings.cfg')

    logger.info("Loading settings from {0}".format(settings_cfg))

    config = configparser.ConfigParser()
    read = config.read(settings_cfg)

    if len(read) == 0 or not config.has_section('main'):
        return

    for name, value in config.items('main'):
        name = name.upper()
        if name in settings:
            if settings.is_default(name) or force:
                setattr(settings, name, json.loads(value))
            elif not settings.is_default(name):
                logger.info("Setting {0} already initialized - skipping".format(name))
        else:
            logger.info("Unknown setting {0} - skipping".format(name))
