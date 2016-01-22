from __future__ import absolute_import, division, print_function

from mock import MagicMock
import matplotlib.pyplot as plt


def renderless_figure():
    # Matplotlib figure that skips the render step, for test speed
    fig = plt.figure()
    fig.canvas.draw = MagicMock()
    plt.close('all')
    return fig
