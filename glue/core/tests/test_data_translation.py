# Tests for the translation infrastrcture between glue Data/Subset/SubsetState
# objects and other data containers registered with glue.

import pytest
import numpy as np
from numpy.testing import assert_equal

from ..component_id import ComponentID
from ..data import Data
from ..data_collection import DataCollection
from ..registry import Registry

from glue.config import data_translator, subset_state_translator
from ..subset import InequalitySubsetState


class FakeDataObject(object):
    array = None
    name = None


class AnotherFakeDataObject(object):
    pass


class CustomSelectionObject(object):
    serialized = None


def setup_module(self):

    @data_translator(FakeDataObject)
    class FakeDataObjectHandler:

        def to_data(self, obj):
            data = Data()
            data[obj.name] = obj.array
            return data

        def to_object(self, data_or_subset):
            cid = data_or_subset.main_components[0]
            obj = FakeDataObject()
            obj.array = data_or_subset[cid]
            obj.name = cid.label
            return obj

    @subset_state_translator('my_subset_translator')
    class FakeSubsetDefinitionTranslator:
        """
        A simple subset translator that only knows how to handle
        InequalitySubsetState and translates it to a custom serialization.
        We include a keyword argument for the translation to make sure it
        gets passed through.
        """

        def to_object(self, subset):

            subset_state = subset.subset_state

            if isinstance(subset_state, InequalitySubsetState):

                if isinstance(subset_state.left, ComponentID):
                    left = '{' + subset_state.left.label + '}'
                else:
                    left = subset_state.left

                if isinstance(subset_state.right, ComponentID):
                    right = '{' + subset_state.right.label + '}'
                else:
                    right = subset_state.right

                c = CustomSelectionObject()
                c.serialized = '{0} {1} {2}'.format(left, subset_state.operator.__name__, right)

                return c

            else:
                raise TypeError("my_subset_translator could not translate "
                                "subset state of type {0}".format(subset_state.__class__.__name__))


def teardown_module(self):
    data_translator.remove(FakeDataObject)
    subset_state_translator.remove('my_subset_translator')


class TestTranslationData:

    def teardown_method(self, method):
        Registry().clear()

    def test_get_object_basic(self):

        data = Data(spam=np.arange(10))

        obj = data.get_object(cls=FakeDataObject)
        assert isinstance(obj, FakeDataObject)
        assert_equal(obj.array, np.arange(10))
        assert_equal(obj.name, 'spam')

    def test_get_object_invalid(self):

        data = Data(spam=np.arange(10))

        with pytest.raises(TypeError) as exc:
            data.get_object(cls=AnotherFakeDataObject)
        assert exc.value.args[0] == 'Could not find a class to translate objects of type Data to AnotherFakeDataObject'

    def test_get_object_explicit_class(self):

        data = Data(x=[1, 2, 3], label='myobj')

        with pytest.raises(ValueError) as exc:
            data.get_object()
        assert exc.value.args[0] == ('Specify the object class to use with cls= - '
                                     'supported classes are:\n\n* pandas.core.frame.DataFrame\n'
                                     '* glue.core.tests.test_data_translation.FakeDataObject')

        obj = data.get_object(cls=FakeDataObject)
        assert isinstance(obj, FakeDataObject)
        assert_equal(obj.array, [1, 2, 3])
        assert_equal(obj.name, 'x')

    def test_get_subset_object(self):

        data = Data(spam=np.arange(10))
        data.add_subset(data.id['spam'] > 4.5, label='subset 1')

        # Check that the following three are equivalent
        for subset_id in [None, 0, 'subset 1']:

            subset = data.get_subset_object(subset_id=subset_id, cls=FakeDataObject)

            assert isinstance(subset, FakeDataObject)
            assert_equal(subset.array, np.arange(5, 10))
            assert subset.name == 'spam'

    def test_get_subset_object_explicit_class(self):

        data = Data(spam=np.arange(10))
        data.add_subset(data.id['spam'] > 4.5, label='subset 1')

        with pytest.raises(ValueError) as exc:
            data.get_subset_object()
        assert exc.value.args[0] == ('Specify the object class to use with cls= - '
                                     'supported classes are:\n\n* pandas.core.frame.DataFrame\n'
                                     '* glue.core.tests.test_data_translation.FakeDataObject')

        subset = data.get_subset_object(subset_id=0, cls=FakeDataObject)

        assert isinstance(subset, FakeDataObject)
        assert_equal(subset.array, np.arange(5, 10))
        assert subset.name == 'spam'

    def test_get_subset_object_invalid(self):

        data = Data(x=np.arange(10), label='myobj')

        with pytest.raises(ValueError) as exc:
            data.get_subset_object(subset_id='subset 2', cls=FakeDataObject)
        assert exc.value.args[0] == "Dataset does not contain any subsets"

        data.add_subset(data.id['x'] > 4.5, label='subset 1')
        data.add_subset(data.id['x'] > 3.5, label='subset 1')

        with pytest.raises(ValueError) as exc:
            data.get_subset_object(subset_id='subset 2', cls=FakeDataObject)
        assert exc.value.args[0] == "No subset found with the label 'subset 2'"

        with pytest.raises(ValueError) as exc:
            data.get_subset_object(cls=FakeDataObject)
        assert exc.value.args[0] == "Several subsets are present, specify which one to retrieve with subset_id= - valid options are:\n\n* 0 or 'subset 1'\n* 1 or 'subset 1_01'"

        # FIXME: currently we disambiguate subset names, but would maybe be
        # better to just raise an error. In any case the test below never
        # works because of the auto-disambiguation.
        # with pytest.raises(ValueError) as exc:
        #     data.get_subset_object(subset_id='subset 1', cls=FakeDataObject)
        # assert exc.value.args[0] == "Several subsets were found with the label 'subset 1', use a numerical index instead"

        subset = data.get_subset_object(subset_id=0, cls=FakeDataObject)

        assert isinstance(subset, FakeDataObject)
        assert_equal(subset.array, np.arange(5, 10))
        assert subset.name == 'x'

    def test_get_selection_basic(self):

        data = Data(x=[1, 2, 3], label='basic')
        data.add_subset(data.id['x'] > 1, label='subset 1')

        # Check that the following three are equivalent
        for subset_id in [None, 0, 'subset 1']:

            result = data.get_selection_definition(subset_id=subset_id,
                                                   format='my_subset_translator')

            assert isinstance(result, CustomSelectionObject)
            assert result.serialized == '{x} gt 1'

    def test_get_selection_invalid(self):

        data = Data(x=[1, 2, 3], label='basic')

        with pytest.raises(ValueError) as exc:
            data.get_selection_definition(subset_id=0)
        assert exc.value.args[0] == "Dataset does not contain any subsets"

        data.add_subset((data.id['x'] > 1) & (data.id['x'] < 3), label='subset 1')

        with pytest.raises(TypeError) as exc:
            data.get_selection_definition(subset_id=0, format='my_subset_translator')
        assert exc.value.args[0] == 'my_subset_translator could not translate subset state of type AndState'

        with pytest.raises(ValueError) as exc:
            data.get_selection_definition(subset_id=0)
        assert exc.value.args[0] == ("Subset state handler format not set - should be one "
                                     "of:\n\n* 'my_subset_translator'")

        with pytest.raises(ValueError) as exc:
            data.get_selection_definition(subset_id=0, format='invalid_translator')
        assert exc.value.args[0] == ("Invalid subset state handler format 'invalid_translator' "
                                     "- should be one of:\n\n* 'my_subset_translator'")

        data.add_subset((data.id['x'] > 1) & (data.id['x'] < 3), label='subset 1')
        data.add_subset((data.id['x'] > 1) & (data.id['x'] < 3), label='subset 3')

        with pytest.raises(ValueError) as exc:
            data.get_selection_definition(subset_id='subset 2', format='invalid_translator')
        assert exc.value.args[0] == "No subset found with the label 'subset 2'"

        with pytest.raises(ValueError) as exc:
            data.get_selection_definition(format='invalid_translator')
        assert exc.value.args[0] == ("Several subsets are present, specify which "
                                     "one to retrieve with subset_id= - valid options "
                                     "are:\n\n* 0 or 'subset 1'\n* 1 or 'subset 1_01'\n* 2 or 'subset 3'")

        # with pytest.raises(ValueError) as exc:
        #     data.get_selection_definition(subset_id='subset 1', format='invalid_translator')
        # assert exc.value.args[0] == "Several subsets were found with the label 'subset 1', use a numerical index instead"


class TestTranslationDataCollection:

    # The main purpose of these tests is to make sure that glue remembers the
    # original class from objects that are added to the data collection.

    def setup_method(self, method):
        self.dc = DataCollection()

    def test_set_and_get_object_basic(self):

        obj = FakeDataObject()
        obj.array = np.arange(10)
        obj.name = 'spam'

        assert len(self.dc) == 0
        self.dc['myobj'] = obj
        assert len(self.dc) == 1
        assert self.dc['myobj'] is self.dc[0]

        data = self.dc['myobj']
        assert isinstance(data, Data)
        assert_equal(data['spam'], np.arange(10))

        # Make sure preferred class is used
        obj_out = self.dc['myobj'].get_object()
        assert isinstance(obj_out, FakeDataObject)
        assert obj_out is not obj
        assert_equal(obj_out.array, np.arange(10))
        assert_equal(obj_out.name, 'spam')

    def test_set_object_invalid(self):

        obj = AnotherFakeDataObject()

        with pytest.raises(TypeError) as exc:
            self.dc['myobj'] = obj
        assert exc.value.args[0] == 'Could not find a class to translate objects of type AnotherFakeDataObject to Data'

    def test_get_subset_object(self):

        obj = FakeDataObject()
        obj.array = np.arange(10)
        obj.name = 'spam'

        self.dc['myobj'] = obj
        self.dc.new_subset_group(subset_state=self.dc['myobj'].id['spam'] > 4.5,
                                 label='subset 1')

        subset = self.dc['myobj'].get_subset_object(subset_id=0)

        # Make sure preferred class is used
        assert isinstance(subset, FakeDataObject)
        assert_equal(subset.array, np.arange(5, 10))
        assert subset.name == 'spam'

    def test_set_duplicate(self):

        # Make sure data gets replaced when re-using a data collection key

        assert len(self.dc) == 0
        self.dc['myobj'] = Data(x=[1, 2, 3])
        assert len(self.dc) == 1
        assert_equal(self.dc[0]['x'], [1, 2, 3])
        self.dc['myobj'] = Data(x=[4, 5, 6])
        assert len(self.dc) == 1
        assert_equal(self.dc[0]['x'], [4, 5, 6])

        # And make sure if there are multiple datasets with the key
        # pre-existing then we get rid of them all

        self.dc.append(Data(x=[7, 8, 9], label='myobj'))
        assert len(self.dc) == 2
        self.dc['myobj'] = Data(x=[1, 2, 3])
        assert len(self.dc) == 1
        assert_equal(self.dc[0]['x'], [1, 2, 3])
