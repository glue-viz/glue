import os

# For backward compatibility, we import get_qapp here
from glue.external.qt import get_qapp


def teardown():
    # can be None if exceptions are raised early during setup -- #323
    if get_qapp is not None:
        app = get_qapp()
        app.exit()

_app = get_qapp()
import atexit
atexit.register(teardown)
