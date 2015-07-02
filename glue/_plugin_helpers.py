# The following funtion is a thin wrapper around iter_entry_tools. The reason it
# is in this separate file is that when making the Mac app, py2app doesn't
# support entry points, so we replace this function with a version that has the
# entry points we want hardcoded. If this function was in glue/main.py, the
# reference to the iter_plugin_entry_points function in load_plugin would be
# evaluated at compile time rather than at runtime, so the patched version
# wouldn't be used.


def iter_plugin_entry_points():
    from pkg_resources import iter_entry_points
    return iter_entry_points(group='glue.plugins', name=None)
