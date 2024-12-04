import os
import sys
import pytest
import subprocess

from glue.config import settings

import numpy as np

from matplotlib.testing.compare import compare_images

__all__ = ['random_with_nan', 'BaseTestExportPython']


def random_with_nan(nsamples, nan_index):
    x = np.random.random(nsamples)
    x[nan_index] = np.nan
    return x


class BaseTestExportPython:

    def assert_same(self, tmpdir, tol=0.1):

        os.chdir(tmpdir.strpath)

        expected = tmpdir.join('expected.png').strpath
        script = tmpdir.join('actual.py').strpath
        actual = tmpdir.join('glue_plot.png').strpath

        self.viewer.axes.figure.savefig(expected)

        self.viewer.export_as_script(script)
        subprocess.call([sys.executable, script])

        msg = compare_images(expected, actual, tol=tol)

        if msg:

            from base64 import b64encode

            print("SCRIPT:")
            with open(script, 'r') as f:
                print(f.read())

            print("EXPECTED:")
            with open(expected, 'rb') as f:
                print(b64encode(f.read()).decode())

            print("ACTUAL:")
            with open(actual, 'rb') as f:
                print(b64encode(f.read()).decode())

            pytest.fail(msg, pytrace=False)

    def test_color_settings(self, tmpdir):
        settings.FOREGROUND_COLOR = '#a51d2d'
        settings.BACKGROUND_COLOR = '#99c1f1'
        self.viewer._update_appearance_from_settings()
        self.assert_same(tmpdir)
        settings.reset_defaults()
        self.viewer._update_appearance_from_settings()
