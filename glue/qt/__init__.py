from __future__ import absolute_import, division, print_function

import os
import atexit

# For backward compatibility, we import get_qapp here
from glue.external.qt import get_qapp


def teardown():
    # can be None if exceptions are raised early during setup -- #323
    if get_qapp is not None:
        app = get_qapp()
        app.exit()

# On ReadTheDocs, we don't want to start up the QApplication instance, since it
# does not support X11 or Xvfb
ON_RTD = os.environ.get('READTHEDOCS', None) == 'True'

if not ON_RTD:
    _app = get_qapp()
    atexit.register(teardown)
