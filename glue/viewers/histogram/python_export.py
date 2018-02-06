from glue.viewers.common.python_export import code, serialize_options


def python_export_histogram_layer(layer, *args):

    if len(layer.mpl_artists) == 0 or not layer.enabled or not layer.visible:
        return [], None

    script = ""
    imports = []

    script += "# Get main data values\n"
    script += "x = layer_data['{0}']\n\n".format(layer._viewer_state.x_att.label)

    script += "# Set up histogram bins\n"
    script += "hist_n_bin = {0}\n".format(layer._viewer_state.hist_n_bin)
    script += "hist_x_min = {0}\n".format(layer._viewer_state.hist_x_min)
    script += "hist_x_max = {0}\n".format(layer._viewer_state.hist_x_max)
    options = dict(alpha=layer.state.alpha,
                   color=layer.state.color,
                   zorder=layer.state.zorder)
    if layer._viewer_state.x_log:
        imports += 'import numpy as np'
        script += "bins = np.logspace(np.log10(hist_x_min), np.log10(hist_x_max), hist_n_bin)\n"
        options['bins'] = code('bins')
    else:
        options['range'] = code('[hist_x_min, hist_x_max]')
        options['bins'] = code('hist_n_bin')

    if layer._viewer_state.normalize:
        options['normed'] = True

    if layer._viewer_state.cumulative:
        options['cumulative'] = True

    script += "\nx = x[(x >= hist_x_min) & (x <= hist_x_max)]\n"

    script += "\nax.hist(x, {0})".format(serialize_options(options))

    return imports, script.strip()
