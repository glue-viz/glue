from __future__ import absolute_import, division, print_function

import os


class MatplotlibBackendSetter(object):
    """
    Import hook to make sure the proper Qt backend is set when importing
    Matplotlib.
    """

    enabled = True

    def find_module(self, mod_name, pth=None):
        if self.enabled and 'matplotlib' in mod_name:
            self.enabled = False
            set_mpl_backend()

    def find_spec(self, name, import_path, target_module=None):
        if self.enabled and name.startswith('matplotlib'):
            self.enabled = False
            set_mpl_backend()


def set_mpl_backend():

    from matplotlib import rcParams, rcdefaults

    # Standardize mpl setup
    rcdefaults()

    # Set default backend to Agg. The Qt and Jupyter glue applications don't
    # use the default backend, so this is just to make sure that importing
    # matplotlib doesn't cause errors related to the MacOSX or Qt backend.
    rcParams['backend'] = 'Agg'

    # Disable key bindings in matplotlib
    for setting in list(rcParams.keys()):
        if setting.startswith('keymap'):
            rcParams[setting] = ''

    # The following is a workaround for the fact that Matplotlib checks the
    # rcParams at import time, not at run-time. This is fixed in Matplotlib>=2.1
    from matplotlib import get_backend
    from matplotlib import backends
    backends.backend = get_backend()

    # Set the MPLBACKEND variable explicitly, because ipykernel uses the lack of
    # MPLBACKEND variable to indicate that it should use its own backend, and
    # this in turn causes some rcParams to be changed, causing test failures
    # etc.
    os.environ['MPLBACKEND'] = 'Agg'
