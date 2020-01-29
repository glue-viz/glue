from glue.viewers.common.python_export import code, serialize_options


def python_export_profile_layer(layer, *args):

    if len(layer.mpl_artists) == 0 or not layer.enabled or not layer.visible:
        return [], None

    script = ""
    imports = ["import numpy as np"]


    script += "# Calculate the profile of the data\n"
    script += "profile_axis = {0}\n".format(layer._viewer_state.x_att_pixel.axis)
    script += "summing_axes = tuple(i for i in range(layer_data.ndim) if i != profile_axis)\n"
    script += "profile_values = layer_data.compute_statistic('{0}', layer_data.find_component_id('{1}'), axis=summing_axes)\n\n".format(
        layer._viewer_state.function, layer.state.attribute.label)

    script += "# Extract the values for the x-axis\n"
    script += "axis_view = [0] * layer_data.ndim\n"
    script += "axis_view[profile_axis] = slice(None)\n"
    script += "profile_x_values = layer_data['{0}', tuple(axis_view)]\n".format(layer._viewer_state.x_att)
    script += "keep = ~np.isnan(profile_values) & ~np.isnan(profile_x_values)\n\n"

    if layer._viewer_state.normalize:
        script += "# Normalize the profile data\n"
        script += "vmax = np.nanmax(profile_values)\n"
        script += "vmin = np.nanmin(profile_values)\n"
        script += "profile_values = (profile_values - vmin)/(vmax - vmin)\n\n"

    script += "# Plot the profile\n"
    plot_options = dict(color=layer.state.color,
                        linewidth=layer.state.linewidth,
                        alpha=layer.state.alpha,
                        zorder=layer.state.zorder)

    script += "ax.plot(profile_x_values[keep], profile_values[keep], '-', {0})\n\n".format(serialize_options(plot_options))

    return imports, script.strip()