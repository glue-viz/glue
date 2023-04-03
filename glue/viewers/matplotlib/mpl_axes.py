import matplotlib.pyplot as plt

from glue.utils.matplotlib import freeze_margins


__all__ = ['update_appearance_from_settings', 'init_mpl']


def set_background_color(axes, color):
    axes.figure.set_facecolor(color)
    axes.patch.set_facecolor(color)


def set_foreground_color(axes, color):
    if hasattr(axes, 'coords'):
        axes.coords.frame.set_color(color)
        axes.coords.frame.set_linewidth(1)
        for coord in axes.coords:
            coord.set_ticks(color=color)
            coord.set_ticklabel(color=color)
            coord.axislabels.set_color(color)
    else:
        for spine in axes.spines.values():
            spine.set_color(color)
        axes.tick_params(which="both",
                         color=color,
                         labelcolor=color)
        axes.xaxis.label.set_color(color)
        axes.yaxis.label.set_color(color)


def set_figure_colors(axes, background, foreground):
    set_background_color(axes, background)
    set_foreground_color(axes, foreground)


def update_appearance_from_settings(axes):
    from glue.config import settings
    set_figure_colors(axes, settings.BACKGROUND_COLOR, settings.FOREGROUND_COLOR)


def init_mpl(figure=None, axes=None, wcs=False, axes_factory=None, projection=None):

    if (axes is not None and figure is not None and
            axes.figure is not figure):
        raise ValueError("Axes and figure are incompatible")

    try:
        from astropy.visualization.wcsaxes import WCSAxesSubplot
    except ImportError:
        WCSAxesSubplot = None

    if axes is not None:
        _axes = axes
        _figure = axes.figure
    else:
        _figure = figure or plt.figure()
        if wcs and WCSAxesSubplot is not None:
            _axes = WCSAxesSubplot(_figure, 111)
            _figure.add_axes(_axes)
        else:
            if axes_factory is None:
                _axes = _figure.add_subplot(1, 1, 1, projection=projection)
            else:
                _axes = axes_factory(_figure)

    freeze_margins(_axes, margins=[1, 0.25, 0.50, 0.25])

    update_appearance_from_settings(_axes)

    return _figure, _axes
