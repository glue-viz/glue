import numpy as np
from glue.core.exceptions import IncompatibleAttribute

__all__ = ['get_mask_with_key_joins']


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

            # TODO: The following is slow because we are looping in Python.
            #       This could be made significantly faster by switching to
            #       C/Cython.

            key_left_all = zip(*key_left_all)
            key_right_all = set(zip(*key_right_all))

            result = [key in key_right_all for key in key_left_all]
            result = np.array(result)

            return result.reshape(data.get_data(cid1_i, view=view).shape)

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
