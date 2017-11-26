from __future__ import absolute_import, division, print_function

import random
from weakref import WeakKeyDictionary

from .core import CallbackProperty

__all__ = ['ChoiceSeparator', 'SelectionCallbackProperty']


class ChoiceSeparator(str):
    pass


class SelectionCallbackProperty(CallbackProperty):

    def __init__(self, default_index=0, **kwargs):
        super(SelectionCallbackProperty, self).__init__(**kwargs)
        self.default_index = default_index
        self._choices = WeakKeyDictionary()
        self._display = WeakKeyDictionary()
        self._force_next_sync = WeakKeyDictionary()

    def __set__(self, instance, value):
        if value is not None and value not in self._choices.get(instance, ()):
            raise ValueError('value {0} is not in valid choices'.format(value))
        super(SelectionCallbackProperty, self).__set__(instance, value)

    def force_next_sync(self, instance):
        self._force_next_sync[instance] = True

    def _get_full_info(self, instance):
        if self._force_next_sync.get(instance, False):
            try:
                return self.__get__(instance), random.random()
            finally:
                self._force_next_sync[instance] = False
        else:
            return self.__get__(instance), self.get_choices(instance), self.get_choice_labels(instance)

    def get_display_func(self, instance):
        return self._display.get(instance, None)

    def set_display_func(self, instance, display):
        self._display[instance] = display
        # selection = self.__get__(instance)
        # self.notify(instance, selection, selection)

    def get_choices(self, instance):
        return self._choices.get(instance, ())

    def get_choice_labels(self, instance):
        display = self._display.get(instance, str)
        labels = []
        for choice in self.get_choices(instance):
            if isinstance(choice, ChoiceSeparator):
                labels.append(str(choice))
            else:
                labels.append(display(choice))
        return labels

    def set_choices(self, instance, choices):
        self._choices[instance] = choices
        self._choices_updated(instance, choices)
        selection = self.__get__(instance)
        self.notify(instance, selection, selection)

    def _choices_updated(self, instance, choices):

        if not choices:
            self.__set__(instance, None)
            return

        selection = self.__get__(instance)

        # We do the following because 'selection in choice' actually compares
        # equality not identity (and we really just care about identity here)
        for choice in choices:
            if selection is choice:
                return

        choices_without_separators = [choice for choice in choices
                                      if not isinstance(choice, ChoiceSeparator)]

        if choices_without_separators:
            try:
                selection = choices_without_separators[self.default_index]
            except IndexError:
                if self.default_index > 0:
                    selection = choices_without_separators[-1]
                else:
                    selection = choices_without_separators[0]
        else:
            selection = None

        self.__set__(instance, selection)
