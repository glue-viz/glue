__all__ = ['get_viewer_tools']


def get_viewer_tools(cls, tools=None, subtools=None):
    """
    Given a viewer class, find all the tools and subtools to include in the
    viewer.

    Parameters
    ----------
    cls : type
        The viewer class for which to look for tools.
    tools : list
        The list to add the tools to - this is modified in-place.
    subtools : dict
        The dictionary to add the subtools to - this is modified in-place.
    """
    if not hasattr(cls, 'inherit_tools'):
        return
    if tools is None:
        tools = []
    if subtools is None:
        subtools = {}
    if cls.inherit_tools:
        for parent_cls in cls.__bases__:
            get_viewer_tools(parent_cls, tools, subtools)
    for tool_id in cls.tools:
        if tool_id not in tools:
            tools.append(tool_id)
    for tool_id in cls.subtools:
        if tool_id not in subtools:
            subtools[tool_id] = []
        for subtool_id in cls.subtools[tool_id]:
            if subtool_id not in subtools[tool_id]:
                subtools[tool_id].append(subtool_id)
    return tools, subtools
