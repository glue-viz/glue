from .... import core


def simple_session():
    collect = core.data_collection.DataCollection()
    hub = core.hub.Hub()
    return core.Session(data_collection=collect, hub=hub,
                        application=None,
                        command_stack=core.CommandStack())
