from glue.config import autolinker

__all__ = ['find_possible_links']


def expand_links(links):
    new_links = []
    if isinstance(links, list):
        for link in links:
            new_links.extend(expand_links(link))
    else:
        new_links.append(links)
    return new_links


def find_possible_links(data_collection):
    """
    Given a `~glue.core.data_collection.DataCollection` object, return a
    dictionary containing possible link suggestions, where the keys are the
    name of the auto-linking plugin, and the values are lists of links.
    """

    suggestions = {}

    for label, function in autolinker:
        links = function(data_collection)
        links = expand_links(links)
        if len(links) > 0:
            suggestions[label] = links

    return suggestions
