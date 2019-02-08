from glue.config import link_wizard

__all__ = ['find_possible_links']


def find_possible_links(data_collection):
    """
    Given a `~glue.core.data_collection.DataCollection` object, return a
    dictionary containing possible link suggestions, where the keys are the
    name of the auto-linking Wizard, and the values are lists of links.
    """

    suggestions = {}

    for label, wizard in link_wizard:
        links = wizard(data_collection)
        if len(links) > 0:
            suggestions[label] = links

    return suggestions
