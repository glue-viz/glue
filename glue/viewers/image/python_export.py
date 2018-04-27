from __future__ import absolute_import, division, print_function

from glue.viewers.common.python_export import code, serialize_options


def python_export_image_layer(layer, *args):

    if not layer.enabled or not layer.visible:
        return [], None

    script = ""
    imports = []

    slices, agg_func, transpose = layer._viewer_state.numpy_slice_aggregation_transpose

    # TODO: implement aggregation, ignore for now

    script += "# Get main data values\n"
    script += "image = layer_data['{0}', {1}]".format(layer.state.attribute, slices)

    if transpose:
        script += ".transpose()"

    script += "\n\n"

    script += "composite.allocate('{0}')\n".format(layer.uuid)

    if layer._viewer_state.color_mode == 'Colormaps':
        color = code('plt.cm.' + layer.state.cmap.name)
    else:
        color = layer.state.color

    options = dict(array=code('image'),
                   clim=(layer.state.v_min, layer.state.v_max),
                   visible=layer.state.visible,
                   zorder=layer.state.zorder,
                   color=color,
                   contrast=layer.state.contrast,
                   bias=layer.state.bias,
                   alpha=layer.state.alpha,
                   stretch=layer.state.stretch)

    script += "composite.set('{0}', {1})\n\n".format(layer.uuid, serialize_options(options))

    return imports, script.strip()


def python_export_image_subset_layer(layer, *args):

    if not layer.enabled or not layer.visible:
        return [], None

    script = ""
    imports = []

    slices, agg_func, transpose = layer._viewer_state.numpy_slice_aggregation_transpose

    # TODO: implement aggregation, ignore for now

    script += "# Get main subset values\n"
    script += "mask = layer_data.to_mask(view={0})\n\n".format(slices)

    imports.append('from glue.utils import color2rgb')
    imports.append('import numpy as np')
    script += "# Convert to RGBA array\n"
    script += "r, g, b = color2rgb('{0}')\n".format(layer.state.color)
    script += "mask = np.dstack((r * mask, g * mask, b * mask, mask * .5))\n"
    script += "mask = (255 * mask).astype(np.uint8)\n"

    if transpose:
        script += ".transpose()"

    script += "\n\n"

    options = dict(origin='lower', interpolation='nearest',
                   vmin=0, vmax=1, aspect=layer._viewer_state.aspect,
                   zorder=layer.state.zorder, alpha=layer.state.alpha)

    script += "imshow(ax, mask, {0})\n".format(serialize_options(options))

    return imports, script.strip()
