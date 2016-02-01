from __future__ import absolute_import, division, print_function

import os

import pkg_resources

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

    try:
        if pkg_resources.resource_exists('glue.icons', icon_name):
            return pkg_resources.resource_filename('glue.icons', icon_name)
        else:
            raise RuntimeError("Icon does not exist: %s" % icon_name)
    except NotImplementedError:  # workaround for mac app
        result = os.path.dirname(__file__)
        return os.path.join(result.replace('site-packages.zip', 'glue'),
                            icon_name)
