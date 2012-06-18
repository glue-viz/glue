import os
import imp

# Set the list of configuration keys that we expect
CONFIG_KEYS = ['extra_clients']

def load_configuration():
    '''
    Read in configuration settings from ~/.glue/config.py
    '''

    # Find absolute path to configuration file
    config_file = os.path.expanduser('~/.glue/config.py')

    # Load the file
    try:
        config = imp.load_source('config', config_file)
    except IOError:
        config = None

    # Populate a configuration dictionary
    config_dict = {}
    for key in CONFIG_KEYS:
        try:
            config_dict[key] = getattr(config, key)
        except AttributeError:
            config_dict[key] = None

    return config_dict
