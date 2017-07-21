from __future__ import absolute_import, division, print_function

from itertools import chain
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

    def __set__(self, instance, value):
        if value is not None and value not in self._choices.get(instance, ()):
            raise ValueError('value {0} is not in valid choices'.format(value))
        super(SelectionCallbackProperty, self).__set__(instance, value)

    def _get_full_info(self, instance):
        return self.__get__(instance), self._choices.get(instance, ())

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

        if selection in choices:
            return

        if self.default_index < 0:
            index = len(choices) + self.default_index
        else:
            index = self.default_index

        index = min(index, len(choices) - 1)

        if isinstance(choices[index], ChoiceSeparator):
            for i in chain(range(index + 1, len(choices)), range(index - 1, -1, -1)):
                if not isinstance(choices[i], ChoiceSeparator):
                    index = i
                    break
            else:
                self.__set__(instance, None)
                return

        self.__set__(instance, choices[index])
