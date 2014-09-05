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


@bball.make_selector
def make_selector(roi, x=None, y=None, **data):

    state = RoiSubsetState()
    state.roi = roi
    state.xatt = x.id
    state.yatt = y.id

    return state


@bball.plot_data
def show_hexbin(axes, x=None, y=None, style=None,
                hit=None, hitrate=None, color=None, bins=None, **kwargs):
    if hitrate:
        return axes.hexbin(x.values, y.values, hit.values,
                           reduce_C_function=lambda x: np.array(x).mean(),
                           cmap=color,
                           gridsize=bins,
                           mincnt=5)
    else:
        return axes.hexbin(x.values, y.values,
                           cmap=color,
                           gridsize=bins,
                           norm=LogNorm(),
                           mincnt=1)


@bball.plot_subset
def show_points(axes, x=None, y=None, style=None, **kwargs):
    return axes.plot(x.values, y.values, 'o',
                     alpha=style.alpha,
                     mec=style.color,
                     mfc=style.color,
                     ms=style.markersize)


@bball.setup
def draw_court(ax):

    c = '#777777'
    opts = dict(fc='none', ec=c, lw=2)
    hoop = Circle((0, 63), radius=9, **opts)
    ax.add_patch(hoop)

    box = Rectangle((-6 * 12, 0), 144, 19 * 12, **opts)
    ax.add_patch(box)

    inner = Arc((0, 19 * 12), 144, 144, theta1=0, theta2=180, **opts)
    ax.add_patch(inner)

    threept = Arc((0, 63), 474, 474, theta1=0, theta2=180, **opts)
    ax.add_patch(threept)

    opts = dict(c=c, lw=2)
    ax.add_line(Line2D([237, 237], [0, 63], **opts))
    ax.add_line(Line2D([-237, -237], [0, 63], **opts))

    ax.set_ylim(0, 400)
    ax.set_aspect('equal', adjustable='datalim')
