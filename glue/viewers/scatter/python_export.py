from glue.viewers.common.python_export import code, serialize_options


def python_export_scatter_layer(layer, *args):

    if len(layer.mpl_artists) == 0 or not layer.enabled or not layer.visible:
        return [], None

    script = ""
    imports = ["import numpy as np"]

    polar = layer._viewer_state.using_polar
    full_sphere = layer._viewer_state.using_full_sphere
    degrees = layer._viewer_state.using_degrees
    theta_formatter = 'ThetaDegreeFormatter' if degrees else 'ThetaRadianFormatter'
    if polar or full_sphere:
        imports += [
            "from glue.core.util import {0}".format(theta_formatter),
        ]
    if polar:
        imports += [
            "from glue.core.util import PolarRadiusFormatter, polar_tick_alignment",
            "from matplotlib.projections.polar import ThetaLocator",
            "from matplotlib.ticker import AutoLocator"
        ]

    script += "layer_handles = []  # for legend"

    script += "# Get main data values\n"
    x_transform_open = "np.radians(" if degrees else ""
    x_transform_close = ")" if degrees else ""
    y_transform_open = "np.radians(" if degrees and full_sphere else ""
    y_transform_close = ")" if degrees and full_sphere else ""
    script += "x = {0}layer_data['{1}']{2}\n".format(x_transform_open, layer._viewer_state.x_att.label, x_transform_close)
    script += "y = {0}layer_data['{1}']{2}\n".format(y_transform_open, layer._viewer_state.y_att.label, y_transform_close)
    if full_sphere:
        script += "x = np.mod(x + np.pi, 2 * np.pi) - np.pi\n"
        if layer._viewer_state.x_min > layer._viewer_state.x_max:
            script += "x = np.negative(x)\n"
        if layer._viewer_state.y_min > layer._viewer_state.y_max:
            script += "y = np.negative(y)\n"
    script += "keep = ~np.isnan(x) & ~np.isnan(y)\n\n"
    if polar:
        script += 'ax.xaxis.set_major_locator(ThetaLocator(AutoLocator()))\n'
        script += 'ax.xaxis.set_major_formatter({0}("{1}"))\n'.format(theta_formatter, layer._viewer_state.x_axislabel)
        script += "for lbl, loc in zip(ax.xaxis.get_majorticklabels(), ax.xaxis.get_majorticklocs()):\n"
        script += "\tlbl.set_horizontalalignment(polar_tick_alignment(loc, {0}))\n\n".format(not degrees)

        script += 'ax.yaxis.set_major_locator(AutoLocator())\n'
        script += 'ax.yaxis.set_major_formatter(PolarRadiusFormatter("{0}"))\n'.format(layer._viewer_state.y_axislabel)
        script += 'for lbl in ax.yaxis.get_majorticklabels():\n'
        script += '\tlbl.set_fontstyle(\'italic\')\n\n'
    elif full_sphere:
        script += 'ax.xaxis.set_major_formatter({0}())\n'.format(theta_formatter)
        if layer._viewer_state.plot_mode != 'lambert':
            script += 'ax.yaxis.set_major_formatter({0}())\n'.format(theta_formatter)

    if layer.state.cmap_mode == 'Linear':

        script += "# Set up colors\n"
        script += "colors = layer_data['{0}']\n".format(layer.state.cmap_att.label)
        script += "cmap_vmin = {0}\n".format(layer.state.cmap_vmin)
        script += "cmap_vmax = {0}\n".format(layer.state.cmap_vmax)
        script += "keep &= ~np.isnan(colors)\n"
        script += "colors = plt.cm.{0}((colors - cmap_vmin) / (cmap_vmax - cmap_vmin))\n\n".format(layer.state.cmap.name)

    if layer.state.size_mode == 'Linear':

        script += "# Set up size values\n"
        script += "sizes = layer_data['{0}']\n".format(layer.state.size_att.label)
        script += "size_vmin = {0}\n".format(layer.state.size_vmin)
        script += "size_vmax = {0}\n".format(layer.state.size_vmax)
        script += "keep &= ~np.isnan(sizes)\n"
        script += "sizes = 30 * (np.clip((sizes - size_vmin) / (size_vmax - size_vmin), 0, 1) * 0.95 + 0.05) * {0}\n\n".format(layer.state.size_scaling)

    if layer.state.markers_visible:
        if layer.state.density_map:

            imports += ["from mpl_scatter_density import ScatterDensityArtist"]
            imports += ["from glue.viewers.scatter.layer_artist import DensityMapLimits, STRETCHES"]
            imports += ["from astropy.visualization import ImageNormalize"]

            script += "density_limits = DensityMapLimits()\n"
            script += "density_limits.contrast = {0}\n\n".format(layer.state.density_contrast)

            options = dict(alpha=layer.state.alpha,
                           zorder=layer.state.zorder,
                           dpi=layer._viewer_state.dpi)

            if layer.state.cmap_mode == 'Fixed':
                options['color'] = layer.state.color
                options['vmin'] = code('density_limits.min')
                options['vmax'] = code('density_limits.max')
                options['norm'] = code("ImageNormalize(stretch=STRETCHES['{0}']())".format(layer.state.stretch))
            else:
                options['c'] = code("layer_data['{0}']".format(layer.state.cmap_att.label))
                options['cmap'] = code("plt.cm.{0}".format(layer.state.cmap.name))
                options['vmin'] = layer.state.cmap_vmin
                options['vmax'] = layer.state.cmap_vmax

            script += "density = ScatterDensityArtist(ax, x, y, {0})\n".format(serialize_options(options))
            script += "ax.add_artist(density)\n\n"

            # legend
            imports += ["from matplotlib.lines import Line2D"]

            options = dict(ms=layer.state.size,
                           alpha=layer.state.alpha,
                           color=layer.state.color)
            script += "layer_handles.append(\n"
            script += "    Line2D([0, ], [0, ],\n"
            script += "           marker='.', linestyle='none',\n"
            script += "           {0}))\n".format(serialize_options(options))

        else:
            if layer._use_plot_artist():
                options = dict(color=layer.state.color,
                               markersize=layer.state.size * layer.state.size_scaling,
                               alpha=layer.state.alpha,
                               zorder=layer.state.zorder,
                               label=layer.layer.label)
                if layer.state.fill:
                    options['mec'] = 'none'
                else:
                    options['mfc'] = 'none'
                script += "plot_artists = ax.plot(x[keep], y[keep], 'o', {0})\n".format(serialize_options(options))
                script += "layer_handles.extend(plot_artists)\n\n"
            else:
                options = dict(alpha=layer.state.alpha,
                               zorder=layer.state.zorder)

                if layer.state.cmap_mode == 'Fixed':
                    options['facecolor'] = layer.state.color
                else:
                    options['c'] = code('colors[keep]')

                if layer.state.size_mode == 'Fixed':
                    options['s'] = code('{0} ** 2'.format(layer.state.size * layer.state.size_scaling))
                else:
                    options['s'] = code('sizes[keep] ** 2')

                if layer.state.fill:
                    options['edgecolor'] = 'none'

                script += "scatter_artist = ax.scatter(x[keep], y[keep], {0})\n".format(serialize_options(options))

                if not layer.state.fill:
                    script += "scatter_artist.set_edgecolors(scatter_artist.get_facecolors())\n"
                    script += "scatter_artist.set_facecolors('none')\n"

                script += "layer_handles.append(scatter_artist)\n\n"

    if layer.state.vector_visible:

        if layer.state.vx_att is not None and layer.state.vy_att is not None:

            imports += ['import numpy as np']

            script += "# Get vector data\n"
            if layer.state.vector_mode == 'Polar':
                script += "angle = layer_data['{0}'][keep]\n".format(layer.state.vx_att.label)
                script += "length = layer_data['{0}'][keep]\n".format(layer.state.vy_att.label)
                script += "vx = length * np.cos(np.radians(angle))\n"
                script += "vy = length * np.sin(np.radians(angle))\n"
            else:
                script += "vx = layer_data['{0}'][keep]\n".format(layer.state.vx_att.label)
                script += "vy = layer_data['{0}'][keep]\n".format(layer.state.vy_att.label)

        if layer.state.vector_arrowhead:
            hw = 3
            hl = 5
        else:
            hw = 1
            hl = 0

        script += 'vmax = np.nanmax(np.hypot(vx, vy))\n\n'

        scale = code('{0} * vmax'.format(10 / layer.state.vector_scaling))

        options = dict(units='width',
                       pivot=layer.state.vector_origin,
                       headwidth=hw, headlength=hl,
                       scale_units='width',
                       scale=scale,
                       angles='xy',
                       alpha=layer.state.alpha,
                       zorder=layer.state.zorder)

        if layer.state.cmap_mode == 'Fixed':
            options['color'] = layer.state.color
        else:
            options['color'] = code('colors[keep]')

        script += "vector_artist = ax.quiver(x[keep], y[keep], vx, vy, {0})\n".format(serialize_options(options))
        script += "layer_handles.append(vector_artist)\n\n"

    if layer.state.xerr_visible or layer.state.yerr_visible:

        if layer.state.xerr_visible and layer.state.xerr_att is not None:
            xerr = code("xerr[keep]")
        else:
            xerr = code("None")

        if layer.state.yerr_visible and layer.state.yerr_att is not None:
            yerr = code("yerr[keep]")
        else:
            yerr = code("None")

        options = dict(fmt='none', xerr=xerr, yerr=yerr,
                       alpha=layer.state.alpha, zorder=layer.state.zorder)

        if layer.state.cmap_mode == 'Fixed':
            options['ecolor'] = layer.state.color
        else:
            options['ecolor'] = code('colors[keep]')

        script += "xerr = layer_data['{0}']\n".format(layer.state.xerr_att.label)
        script += "yerr = layer_data['{0}']\n".format(layer.state.yerr_att.label)
        script += "keep &= ~np.isnan(xerr) & ~np.isnan(yerr)\n"
        script += "error_artist = ax.errorbar(x[keep], y[keep], {0})\n".format(serialize_options(options))
        script += "layer_handles.append(error_artist)\n\n"

    if layer.state.line_visible:

        options = dict(color=layer.state.color,
                       linewidth=layer.state.linewidth,
                       linestyle=layer.state.linestyle,
                       alpha=layer.state.alpha,
                       zorder=layer.state.zorder)
        if layer.state.cmap_mode == 'Fixed':
            script += "ax.plot(x[keep], y[keep], '-', {0})\n\n".format(serialize_options(options))
        else:
            options['c'] = code('colors')
            imports.append("from glue.viewers.scatter.layer_artist import plot_colored_line")
            script += "line_collection = plot_colored_line(ax, x, y, {0})\n".format(serialize_options(options))
            script += "layer_handles.append(line_collection)\n\n"

    script += "legend_handles.append(tuple(layer_handles))\n"
    script += "legend_labels.append(layer_data.label)\n\n"
    return imports, script.strip()
