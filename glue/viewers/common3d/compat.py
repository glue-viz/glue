import uuid

from ..scatter3d.layer_state import ScatterLayerState
from ..volume3d.layer_state import VolumeLayerState3D

STATE_CLASS = {}
STATE_CLASS['ScatterLayerArtist'] = ScatterLayerState
STATE_CLASS['VolumeLayerArtist'] = VolumeLayerState3D


def update_3d_viewer_state(rec, context):
    """
    Given viewer session information, make sure the session information is
    compatible with the current version of the viewers, and if not, update
    the session information in-place.
    """

    if '_protocol' not in rec:

        rec.pop('properties')

        rec['state'] = {}
        rec['state']['values'] = rec.pop('options')

        layer_states = []

        for layer in rec['layers']:
            state_id = str(uuid.uuid4())
            state_cls = STATE_CLASS[layer['_type'].split('.')[-1]]
            state = state_cls(layer=context.object(layer.pop('layer')))
            properties = set(layer.keys()) - set(['_type'])
            for prop in sorted(properties, key=state._update_priority, reverse=True):
                value = layer.pop(prop)
                value = context.object(value)
                if isinstance(value, str) and value == 'fixed':
                    value = 'Fixed'
                if isinstance(value, str) and value == 'linear':
                    value = 'Linear'
                setattr(state, prop, value)
            context.register_object(state_id, state)
            layer['state'] = state_id
            layer_states.append(state)

        list_id = str(uuid.uuid4())
        context.register_object(list_id, layer_states)
        rec['state']['values']['layers'] = list_id

        rec['state']['values']['visible_axes'] = rec['state']['values'].pop('visible_box')
