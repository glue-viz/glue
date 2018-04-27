from __future__ import absolute_import, division, print_function

import uuid

import numpy as np
from glue.viewers.image.state import ImageLayerState, ImageSubsetLayerState
from glue.viewers.scatter.state import ScatterLayerState

STATE_CLASS = {}
STATE_CLASS['ImageLayerArtist'] = ImageLayerState
STATE_CLASS['ScatterLayerArtist'] = ScatterLayerState
STATE_CLASS['SubsetImageLayerArtist'] = ImageSubsetLayerState


class DS9Compat(object):

    @classmethod
    def __setgluestate__(cls, rec, context):
        result = cls()
        for k, v in rec.items():
            setattr(result, k, v)
        return result


def update_image_viewer_state(rec, context):
    """
    Given viewer session information, make sure the session information is
    compatible with the current version of the viewers, and if not, update
    the session information in-place.
    """

    if '_protocol' not in rec:

        # Note that files saved with protocol < 1 have bin settings saved per
        # layer but they were always restricted to be the same, so we can just
        # use the settings from the first layer

        rec['state'] = {}
        rec['state']['values'] = {}

        # TODO: could generalize this into a mapping
        properties = rec.pop('properties')
        viewer_state = rec['state']['values']
        viewer_state['color_mode'] = 'st__Colormaps'
        viewer_state['reference_data'] = properties['data']

        data = context.object(properties['data'])

        # TODO: add an id method to unserializer

        x_index = properties['slice'].index('x')
        y_index = properties['slice'].index('y')

        viewer_state['x_att_world'] = str(uuid.uuid4())
        context.register_object(viewer_state['x_att_world'], data.world_component_ids[x_index])

        viewer_state['y_att_world'] = str(uuid.uuid4())
        context.register_object(viewer_state['y_att_world'], data.world_component_ids[y_index])

        viewer_state['x_att'] = str(uuid.uuid4())
        context.register_object(viewer_state['x_att'], data.pixel_component_ids[x_index])

        viewer_state['y_att'] = str(uuid.uuid4())
        context.register_object(viewer_state['y_att'], data.pixel_component_ids[y_index])

        viewer_state['x_min'] = -0.5
        viewer_state['x_max'] = data.shape[1] - 0.5
        viewer_state['y_min'] = -0.5
        viewer_state['y_max'] = data.shape[0] - 0.5

        viewer_state['aspect'] = 'st__equal'

        # Slicing with cubes
        viewer_state['slices'] = [s if np.isreal(s) else 0 for s in properties['slice']]

        # RGB mode
        for layer in rec['layers'][:]:
            if layer['_type'].split('.')[-1] == 'RGBImageLayerArtist':
                for icolor, color in enumerate('rgb'):
                    new_layer = {}
                    new_layer['_type'] = 'glue.viewers.image.layer_artist.ImageLayerArtist'
                    new_layer['layer'] = layer['layer']
                    new_layer['attribute'] = layer[color]
                    new_layer['norm'] = layer[color + 'norm']
                    new_layer['zorder'] = layer['zorder']
                    new_layer['visible'] = layer['color_visible'][icolor]
                    new_layer['color'] = color
                    rec['layers'].append(new_layer)
                rec['layers'].remove(layer)
                viewer_state['color_mode'] = 'st__One color per layer'

        layer_states = []

        for layer in rec['layers']:

            state_id = str(uuid.uuid4())
            state_cls = STATE_CLASS[layer['_type'].split('.')[-1]]
            state = state_cls(layer=context.object(layer.pop('layer')))
            for prop in ('visible', 'zorder'):
                value = layer.pop(prop)
                value = context.object(value)
                setattr(state, prop, value)
            if 'attribute' in layer:
                state.attribute = context.object(layer['attribute'])
            else:
                state.attribute = context.object(properties['attribute'])
            if 'norm' in layer:
                norm = context.object(layer['norm'])
                state.bias = norm.bias
                state.contrast = norm.contrast
                state.stretch = norm.stretch
                if norm.clip_hi is not None:
                    state.percentile = norm.clip_hi
                else:
                    if norm.vmax is not None:
                        state.v_min = norm.vmin
                        state.v_max = norm.vmax
                        state.percentile = 'Custom'
            if 'color' in layer:
                state.global_sync = False
                state.color = layer['color']
            context.register_object(state_id, state)
            layer['state'] = state_id
            layer_states.append(state)

        list_id = str(uuid.uuid4())
        context.register_object(list_id, layer_states)
        rec['state']['values']['layers'] = list_id
