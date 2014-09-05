from glue import custom_viewer

from matplotlib.colors import LogNorm

bball = custom_viewer('Shot Plot',
                      x='att(x)',
                      y='att(y)')


@bball.plot_data
def show_hexbin(axes, x=None, y=None, style=None, **kwargs):
    return axes.hexbin(x.values, y.values,
                       cmap='Purples',
                       gridsize=40,
                       norm=LogNorm(),
                       mincnt=1)


@bball.plot_subset
def show_points(axes, x=None, y=None, style=None, **kwargs):
    return axes.plot(x.values, y.values, 'o',
                     alpha=style.alpha,
                     mec=style.color,
                     mfc=style.color,
                     ms=style.markersize)
