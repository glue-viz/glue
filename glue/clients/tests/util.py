import matplotlib.pyplot as plt
from mock import MagicMock


def renderless_figure():
    # Matplotlib figure that skips the render step, for test speed
    fig = plt.figure()
    fig.canvas.draw = MagicMock()
    plt.close('all')
    return fig
