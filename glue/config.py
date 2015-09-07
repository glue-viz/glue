"""
Objects used to configure Glue at runtime.
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import imp

from .core.config import *
from .qt.config import *
from .plugins.config import *

__all__ = ['load_configuration']

from .core import config
__all__ += config.__all__
del config

from .qt import config
__all__ += config.__all__
del config

from .plugins import config
__all__ += config.__all__
del config


CFG_DIR = os.path.join(os.path.expanduser('~'), '.glue')



def load_configuration(search_path=None):
    ''' Find and import a config.py file

    Returns:

       The module object

    Raises:

       Exception, if no module was found
    '''
    search_order = search_path or _default_search_order()
    result = imp.new_module('config')

    for config_file in search_order:
        dir = os.path.dirname(config_file)
        try:
            sys.path.append(dir)
            config = imp.load_source('config', config_file)
            result = config
        except IOError:
            pass
        except Exception as e:
            raise type(e)("Error loading config file %s:\n%s" % (config_file, e), sys.exc_info()[2])
        finally:
            sys.path.remove(dir)

    return result


def _default_search_order():
    """
    The default configuration file search order:

       * current working directory
       * environ var GLUERC
       * HOME/.glue/config.py
       * Glue's own default config
    """

    from . import config

    search_order = [os.path.join(os.getcwd(), 'config.py')]
    if 'GLUERC' in os.environ:
        search_order.append(os.environ['GLUERC'])
    search_order.append(os.path.join(config.CFG_DIR, 'config.py'))
    return search_order[::-1]
