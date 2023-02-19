import sys

if sys.version_info >= (3, 9):
    import importlib.resources as importlib_resources
else:
    import importlib_resources


__all__ = ['icon_path']


def icon_path(icon_name, icon_format='png'):
    """
    Return the absolute path to an icon

    Parameters
    ----------
    icon_name : str
       Name of icon, without extension or directory prefix
    icon_format : str, optional
        Can be either 'png' or 'svg'

    Returns
    -------
    path : str
      Full path to icon
    """

    icon_name += '.{0}'.format(icon_format)

    icon_file = importlib_resources.files("glue") / "icons" / icon_name
    if icon_file.is_file():
        return str(icon_file)
    else:
        raise RuntimeError("Icon does not exist: %s" % icon_name)
