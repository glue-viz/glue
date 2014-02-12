import matplotlib.pyplot as plt


def renderless_figure():
    # Matplotlib figure that skips the render step, for test speed
    fig = plt.figure()
    fig.canvas.draw = lambda: 0
    plt.close('all')
    return fig
