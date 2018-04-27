from __future__ import absolute_import, division, print_function

import uuid

from .state import ScatterLayerState

STATE_CLASS = {}
STATE_CLASS['ScatterLayerArtist'] = ScatterLayerState


def update_scatter_viewer_state(rec, context):
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
        viewer_state['x_min'] = properties['xmin']
        viewer_state['x_max'] = properties['xmax']
        viewer_state['y_min'] = properties['ymin']
        viewer_state['y_max'] = properties['ymax']
        viewer_state['x_log'] = properties['xlog']
        viewer_state['y_log'] = properties['ylog']
        viewer_state['x_att'] = properties['xatt']
        viewer_state['y_att'] = properties['yatt']

        layer_states = []

        for layer in rec['layers']:
            state_id = str(uuid.uuid4())
            state_cls = STATE_CLASS[layer['_type'].split('.')[-1]]
            state = state_cls(layer=context.object(layer.pop('layer')))
            for prop in ('visible', 'zorder'):
                value = layer.pop(prop)
                value = context.object(value)
                setattr(state, prop, value)
            context.register_object(state_id, state)
            layer['state'] = state_id
            layer_states.append(state)
            layer.pop('lo', None)
            layer.pop('hi', None)
            layer.pop('nbins', None)
            layer.pop('xlog', None)

        list_id = str(uuid.uuid4())
        context.register_object(list_id, layer_states)
        rec['state']['values']['layers'] = list_id
