from glue import custom_viewer

from matplotlib.colors import LogNorm

bball = custom_viewer('Shot Plot',
                      x='att(x)',
                      y='att(y)')


@bball.plot_data
def show_hexbin(axes, x, y):
    axes.hexbin(x, y,
                cmap='Purples',
                gridsize=40,
                norm=LogNorm(),
                mincnt=1)


@bball.plot_subset
def show_points(axes, x, y, style):
    axes.plot(x, y, 'o',
              alpha=style.alpha,
              mec=style.color,
              mfc=style.color,
              ms=style.markersize)
