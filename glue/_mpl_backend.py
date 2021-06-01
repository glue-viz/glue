import os
import numpy as np


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

    # Set the MPLBACKEND variable explicitly, because ipykernel uses the lack of
    # MPLBACKEND variable to indicate that it should use its own backend, and
    # this in turn causes some rcParams to be changed, causing test failures
    # etc.
    os.environ['MPLBACKEND'] = 'Agg'

    # Explicitly switch backend
    from matplotlib.pyplot import switch_backend
    switch_backend('agg')

    # We override the datetime64 units.registry entry to use our class. We do
    # this here to make sure that rcdefaults hasn't reset this.
    import matplotlib.units as units
    from glue.utils.matplotlib import Datetime64Converter
    units.registry[np.datetime64] = Datetime64Converter()
