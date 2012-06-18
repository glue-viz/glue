import os
import imp

CONFIG_KEYS = ['extra_clients']

def load_configuration():
    '''
    Read in configuration settings from a config.py file.

    Search order:

     * current working directory
     * environ var GLUERC
     * HOME/.glue/config.py
    '''
    config = _load_config_file()

    # Populate a configuration dictionary
    config_dict = {}
    for key in CONFIG_KEYS:
        try:
            config_dict[key] = getattr(config, key)
        except AttributeError:
            config_dict[key] = None

    return config_dict

def _load_config_file():
    ''' Find and import a config.py file

    Returns
    -------
    The module object, or None if no file found
    '''
    search_order = [os.path.join(os.getcwd(), 'config.py')]
    if 'GLUERC' in os.environ:
        search_order.append(os.environ['GLUERC'])
    search_order.append(os.path.expanduser('~/.glue/config.py'))

    config = None
    for config_file in search_order:
        # Load the file
        try:
            config = imp.load_source('config', config_file)
            return config
        except IOError:
            pass