import uuid

from glue.viewers.common.state import LayerState

STATE_CLASS = {}
STATE_CLASS['TableLayerArtist'] = LayerState


def update_table_viewer_state(rec, context):
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

        _ = rec.pop('properties')

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

        list_id = str(uuid.uuid4())
        context.register_object(list_id, layer_states)
        rec['state']['values']['layers'] = list_id
