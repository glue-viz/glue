import os
import sys
import imp

def load_configuration(search_path = None):
    ''' Find and import a config.py file

    Returns:

       The module object

    Raises:

       Exception, if no module was found
    '''
    search_order = search_path or _default_search_order()

    for config_file in search_order:
        try:
            config = imp.load_source('config', config_file)
            return config
        except IOError:
            pass
        except Exception as e:
            raise Exception("Error loading config file %s:\n%s" %
                            (config_file, e))

    raise Exception("Could not find a valid glue config file")


def _default_search_order():
    """
    The default configuration file search order:

       * current working directory
       * environ var GLUERC
       * HOME/.glue/config.py
       * Glue's own default_config.py
    """
    current_module = sys.modules['glue'].__path__[0]
    search_order = [os.path.join(os.getcwd(), 'config.py')]
    if 'GLUERC' in os.environ:
        search_order.append(os.environ['GLUERC'])
    search_order.append(os.path.expanduser('~/.glue/config.py'))
    search_order.append(os.path.join(current_module, 'default_config.py'))
    return search_order
