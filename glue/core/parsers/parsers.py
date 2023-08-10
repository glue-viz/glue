# Functions used to parse data and links from Python command-line

import numpy as np

from glue.config import cli_parser
from glue.core import BaseData, Data

__all__ = []


@cli_parser(dict)
def _parse_data_dict(data, label):
    result = Data(label=label)
    for label, component in data.items():
        result.add_component(component, label)
    return [result]


@cli_parser(np.recarray)
def _parse_data_recarray(data, label):
    kwargs = dict((n, data[n]) for n in data.dtype.names)
    return [Data(label=label, **kwargs)]


@cli_parser(BaseData)
def _parse_data_glue_data(data, label):
    if isinstance(data, Data):
        data.label = label
    return [data]


@cli_parser(np.ndarray)
def _parse_data_numpy(data, label):
    return [Data(**{label: data, 'label': label})]


@cli_parser(list)
def _parse_data_list(data, label):
    return [Data(**{label: data, 'label': label})]


@cli_parser(str)
def _parse_data_path(path, label):
    from glue.core.data_factories import load_data, as_list

    data = load_data(path)
    for d in as_list(data):
        d.label = label
    return as_list(data)


def parse_data(data, label):

    # First try new data translation layer

    from glue.config import data_translator

    try:
        handler, preferred = data_translator.get_handler_for(data)
    except TypeError:
        pass
    else:
        data = handler.to_data(data)
        data.label = label
        data._preferred_translation = preferred
        return [data]

    # Then try legacy 'cli_parser' infrastructure

    for item in cli_parser:

        data_class = item.data_class
        parser = item.parser
        if isinstance(data, data_class):
            try:
                return parser(data, label)
            except Exception as e:
                raise ValueError("Invalid format for data '%s'\n\n%s" %
                                 (label, e))

    raise TypeError("Invalid data description: %s" % data)


def parse_links(dc, links):
    from glue.core.link_helpers import MultiLink
    from glue.core import ComponentLink

    data = dict((d.label, d) for d in dc)
    result = []

    def find_cid(s):
        dlabel, clabel = s.split('.')
        d = data[dlabel]
        c = d.find_component_id(clabel)
        if c is None:
            raise ValueError("Invalid link (no component named %s)" % s)
        return c

    for link in links:
        f, t = link[0:2]  # from and to component names
        u = u2 = None
        if len(link) >= 3:  # forward translation function
            u = link[2]
        if len(link) == 4:  # reverse translation function
            u2 = link[3]

        # component names -> component IDs
        if isinstance(f, str):
            f = [find_cid(f)]
        else:
            f = [find_cid(item) for item in f]

        if isinstance(t, str):
            t = find_cid(t)
            result.append(ComponentLink(f, t, u))
        else:
            t = [find_cid(item) for item in t]
            result += MultiLink(f, t, u, u2)

    return result
