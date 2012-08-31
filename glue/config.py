import os
import sys
import imp


class ConfigObject(object):

    def __init__(self):
        from .qt.widgets.scatter_widget import ScatterWidget
        from .qt.widgets.image_widget import ImageWidget
        from .qt.widgets.histogram_widget import HistogramWidget
        from .core import link_helpers

        self.qt_clients = [ScatterWidget, ImageWidget, HistogramWidget]
        self.link_functions = list(link_helpers.__LINK_FUNCTIONS__)

    def merge_module(self, module):
        """Import public attributes from module into instance attributes

        :param module: the module to add
        """
        for name, obj in vars(module).items():
            if name.startswith('_'):
                continue
            self.__setattr__(name, obj)


def load_configuration(search_path=None):
    ''' Find and import a config.py file

    Returns:

       The module object

    Raises:

       Exception, if no module was found
    '''
    search_order = search_path or _default_search_order()
    result = ConfigObject()

    for config_file in search_order:
        try:
            config = imp.load_source('config', config_file)
            result.merge_module(config)
            #return config
        except IOError:
            pass
        except Exception as e:
            print e, config_file
            raise Exception("Error loading config file %s:\n%s" %
                            (config_file, e))

    return result


def _default_search_order():
    """
    The default configuration file search order:

       * current working directory
       * environ var GLUERC
       * HOME/.glue/config.py
       * Glue's own default config
    """
    current_module = sys.modules['glue'].__path__[0]
    search_order = [os.path.join(os.getcwd(), 'config.py')]
    if 'GLUERC' in os.environ:
        search_order.append(os.environ['GLUERC'])
    search_order.append(os.path.expanduser('~/.glue/config.py'))
    return search_order[::-1]
