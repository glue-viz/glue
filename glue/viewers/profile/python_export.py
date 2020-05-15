from glue.viewers.common.python_export import serialize_options
from glue.core import Subset


def python_export_profile_layer(layer, *args):

    if len(layer.mpl_artists) == 0 or not layer.enabled or not layer.visible:
        return [], None

    script = ""
    imports = ["import numpy as np"]

    script += "# Calculate the profile of the data\n"
    script += "profile_axis = {0}\n".format(layer._viewer_state.x_att_pixel.axis)
    script += "collapsed_axes = tuple(i for i in range(layer_data.ndim) if i != profile_axis)\n"
    if isinstance(layer.state.layer, Subset):
        script += "base_data = layer_data.data\n"
        script += "cid = base_data.find_component_id('{0}')\n".format(layer.state.attribute.label)
        script += "profile_values = base_data.compute_statistic('{0}', cid, axis=collapsed_axes, subset_state=layer_data.subset_state)\n\n".format(layer._viewer_state.function)
    else:
        script += "cid = layer_data.find_component_id('{0}')\n".format(layer.state.attribute.label)
        script += "profile_values = layer_data.compute_statistic('{0}', cid, axis=collapsed_axes)\n\n".format(layer._viewer_state.function)

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
                        zorder=layer.state.zorder,
                        drawstyle='steps-mid')

    script += "handle,  = ax.plot(profile_x_values[keep], profile_values[keep], '-', {0})\n".format(serialize_options(plot_options))
    script += "legend_handles.append(handle)\n"
    script += "legend_labels.append(layer_data.label)\n\n"

    return imports, script.strip()
