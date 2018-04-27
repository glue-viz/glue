from __future__ import absolute_import, division, print_function


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

    try:
        from qtpy import PYQT5  # noqa
    except Exception:
        # If Qt isn't available, we don't have to worry about
        # setting the backend
        return

    from matplotlib import rcParams, rcdefaults

    # standardize mpl setup
    rcdefaults()

    rcParams['backend'] = 'Qt5Agg'

    # disable key bindings in matplotlib
    for setting in list(rcParams.keys()):
        if setting.startswith('keymap'):
            rcParams[setting] = ''

    # The following is a workaround for the fact that Matplotlib checks the
    # rcParams at import time, not at run-time. I have opened an issue with
    # Matplotlib here: https://github.com/matplotlib/matplotlib/issues/5513
    from matplotlib import get_backend
    from matplotlib import backends
    backends.backend = get_backend()
