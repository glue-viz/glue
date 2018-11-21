import numpy as np
from glue.core.exceptions import IncompatibleAttribute

__all__ = ['get_mask_with_key_joins']


def concatenate_arrays(*arrays):
    """
    Given N arrays of size M, return an array of size M where each item is
    the concatenation of the bytes of the orginal arrays for the respective
    item.
    """

    # Find the total size of an item in the final array
    total_size = sum(x.dtype.itemsize for x in arrays)

    # Set up the final array
    buffer = np.zeros(len(arrays[0]) * total_size, dtype=np.byte)

    # Now view this array as a structured array
    colnames = ['{0:x}'.format(x) for x in range(len(arrays))]
    dtype = list(zip(colnames, [x.dtype for x in arrays]))
    buffer_as_struct = buffer.view(dtype)

    # Finally, fill the array column by column
    for letter, array in zip(colnames, arrays):
        buffer_as_struct[letter] = array

    # And view as an array of size M
    return buffer_as_struct.view('S{0}'.format(total_size))


def get_mask_with_key_joins(data, key_joins, subset_state, view=None):
    """
    Given a dataset and a subset state, check whether the subset state
    can be translated to the current dataset via key joins.

    Note that this does not try simply applying the subset state to the
    dataset, as it is assumed this has been tried first.
    """

    for other, (cid1, cid2) in key_joins.items():

        if getattr(other, '_recursing', False):
            continue

        try:
            data._recursing = True
            mask_right = other.get_mask(subset_state)
        except IncompatibleAttribute:
            continue
        finally:
            data._recursing = False

        if len(cid1) == 1 and len(cid2) == 1:

            key_left = data.get_data(cid1[0], view=view)
            key_right = other.get_data(cid2[0], view=mask_right)
            mask = np.in1d(key_left.ravel(), key_right.ravel())

            return mask.reshape(key_left.shape)

        elif len(cid1) == len(cid2):

            key_left_all = []
            key_right_all = []

            for cid1_i, cid2_i in zip(cid1, cid2):
                key_left_all.append(data.get_data(cid1_i, view=view).ravel())
                key_right_all.append(other.get_data(cid2_i, view=mask_right).ravel())

            key_left_all = concatenate_arrays(*key_left_all)
            key_right_all = concatenate_arrays(*key_right_all)

            mask = np.in1d(key_left_all, key_right_all)

            return mask.reshape(data.get_data(cid1_i, view=view).shape)

        elif len(cid1) == 1:

            key_left = data.get_data(cid1[0], view=view).ravel()
            mask = np.zeros_like(key_left, dtype=bool)
            for cid2_i in cid2:
                key_right = other.get_data(cid2_i, view=mask_right).ravel()
                mask |= np.in1d(key_left, key_right)

            return mask.reshape(data.get_data(cid1[0], view=view).shape)

        elif len(cid2) == 1:

            key_right = other.get_data(cid2[0], view=mask_right).ravel()
            mask = np.zeros_like(data.get_data(cid1[0], view=view).ravel(), dtype=bool)
            for cid1_i in cid1:
                key_left = data.get_data(cid1_i, view=view).ravel()
                mask |= np.in1d(key_left, key_right)

            return mask.reshape(data.get_data(cid1[0], view=view).shape)

        else:

            raise Exception("Either the number of components in the key join sets "
                            "should match, or one of the component sets should ",
                            "contain a single component.")

    raise IncompatibleAttribute
