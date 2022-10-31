from glue.viewers.common.python_export import code, serialize_options


def python_export_image_layer(layer, *args):

    if not layer.enabled or not layer.visible:
        return [], None

    script = ""
    imports = ["from glue.viewers.image.state import get_sliced_data_maker"]
    imports += ["import matplotlib.patches as mpatches"]

    slices, agg_func, transpose = layer._viewer_state.numpy_slice_aggregation_transpose

    # TODO: implement aggregation, ignore for now

    script += "# Define a function that will get a fixed resolution buffer\n"

    options = {'data': code('layer_data'),
               'x_axis': layer._viewer_state.x_att.axis,
               'y_axis': layer._viewer_state.y_att.axis,
               'slices': slices,
               'target_cid': code("layer_data.id['{0}']".format(layer.state.attribute))}

    if transpose:
        options['transpose'] = True

    script += "array_maker = get_sliced_data_maker({0})\n\n".format(serialize_options(options))

    script += "composite.allocate('{0}')\n".format(layer.uuid)

    if layer._viewer_state.color_mode == 'Colormaps':
        script += "composite.mode = 'colormap'\n"
    else:
        script += "composite.mode = 'color'\n"

    options = dict(array=code('array_maker'),
                   clim=(layer.state.v_min, layer.state.v_max),
                   visible=layer.state.visible,
                   zorder=layer.state.zorder,
                   color=layer.state.color,
                   cmap=code('plt.cm.' + layer.state.cmap.name),
                   contrast=layer.state.contrast,
                   bias=layer.state.bias,
                   alpha=layer.state.alpha,
                   stretch=layer.state.stretch)

    script += "composite.set('{0}', {1})\n".format(layer.uuid, serialize_options(options))

    if layer._viewer_state.color_mode == 'Colormaps':
        imports += ["from glue.utils.matplotlib import ColormapPatchHandler"]
        script += "handle = mpatches.Patch(color='{0}')\n".format(layer.state.color)
        script += "handler = ColormapPatchHandler(" + code('plt.cm.' + layer.state.cmap.name) + ")\n"

        script += "legend_handles.append(handle)\n"
        script += "legend_handler_dict[handle] = handler\n"
        script += "legend_labels.append(layer_data.label)\n"
    else:
        options = dict(color=layer.state.color,
                       alpha=layer.state.alpha)
        script += "handle = mpatches.Patch({0})\n".format(serialize_options(options))
        script += "legend_handles.append(handle)\n"
        script += "legend_labels.append(layer_data.label)\n"

    script += "\n"

    return imports, script.strip()


def python_export_image_subset_layer(layer, *args):

    if not layer.enabled or not layer.visible:
        return [], None

    script = ""
    imports = ["from glue.viewers.image.state import get_sliced_data_maker"]

    slices, agg_func, transpose = layer._viewer_state.numpy_slice_aggregation_transpose

    # TODO: implement aggregation, ignore for now

    script += "# Define a function that will get a fixed resolution buffer of the mask\n"

    options = {'data': code('layer_data'),
               'x_axis': layer._viewer_state.x_att.axis,
               'y_axis': layer._viewer_state.y_att.axis,
               'slices': slices}

    if transpose:
        options['transpose'] = True

    script += "array_maker = get_sliced_data_maker({0})\n\n".format(serialize_options(options))

    script += "\n\n"

    options = dict(origin='lower', interpolation='nearest', color=layer.state.color,
                   vmin=0, vmax=1, aspect=layer._viewer_state.aspect,
                   zorder=layer.state.zorder, alpha=layer.state.alpha)

    script += "imshow(ax, {0}, {1})\n".format(code('array_maker'), serialize_options(options))

    # legend
    options = dict(color=layer.state.color,
                   alpha=layer.state.alpha)
    script += "handle = mpatches.Patch({0})\n".format(serialize_options(options))
    script += "legend_handles.append(handle)\n"
    script += "legend_labels.append(layer_data.label)\n"
    script += "\n"

    return imports, script.strip()
