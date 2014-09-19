from glue import custom_viewer
from glue.core.subset import RoiSubsetState

from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.lines import Line2D
import numpy as np

bball = custom_viewer('Shot Plot',
                      x='att(x)',
                      y='att(y)',
                      bins=(10, 100),
                      hitrate=False,
                      color=['Reds', 'Purples'],
                      hit='att(shot_made)')


@bball.select
def select(roi, x, y):
    return roi.contains(x, y)


@bball.plot_data
def show_hexbin(axes, x, y,
                hit, hitrate, color, bins):
    if hitrate:
        axes.hexbin(x, y, hit,
                    reduce_C_function=lambda x: np.array(x).mean(),
                    cmap=color,
                    gridsize=bins,
                    mincnt=5)
    else:
        axes.hexbin(x, y,
                    cmap=color,
                    gridsize=bins,
                    norm=LogNorm(),
                    mincnt=1)


@bball.plot_subset
def show_points(axes, x, y, style):
    axes.plot(x, y, 'o',
              alpha=style.alpha,
              mec=style.color,
              mfc=style.color,
              ms=style.markersize)


@bball.setup
def draw_court(axes):

    c = '#777777'
    opts = dict(fc='none', ec=c, lw=2)
    hoop = Circle((0, 63), radius=9, **opts)
    axes.add_patch(hoop)

    box = Rectangle((-6 * 12, 0), 144, 19 * 12, **opts)
    axes.add_patch(box)

    inner = Arc((0, 19 * 12), 144, 144, theta1=0, theta2=180, **opts)
    axes.add_patch(inner)

    threept = Arc((0, 63), 474, 474, theta1=0, theta2=180, **opts)
    axes.add_patch(threept)

    opts = dict(c=c, lw=2)
    axes.add_line(Line2D([237, 237], [0, 63], **opts))
    axes.add_line(Line2D([-237, -237], [0, 63], **opts))

    axes.set_ylim(0, 400)
    axes.set_aspect('equal', adjustable='datalim')
