from collections import namedtuple

Session = namedtuple('Session',
                     'data_collection application command_stack hub')
