Building Custom Subsets with the Terminal Window
================================================

The built-in terminal window gives you the ability to inspect and edit
data objects in Glue. This tutorial demonstrates how to use it to build
custom subsets that would be cumbersome to define manually.


.. note:: The terminal window requires that you have IPython > v0.12 on your system

We are using the example data for this tutorial. We also define a few
subsets to play with. Our setup looks like this.

.. image:: subset_01.png
   :width: 60%

Click the arrow button in the lower left to open the terminal window.

The individual data sets can be found in data_collection.data::

    >>> data_collection.data
    [<glue.core.data.Data at 0x112059dd0>]
    >>> data = data_collection.data[0]

You can also get the data object by indexing into DataCollection::

    >>> data is data_collection[0]
    True

Assign variables to the two subsets defined for this data::

    >>> high_av, ngc1333 = data.subsets

Let's also grab a few of the components in the data::

    >>> ra = data.find_component_id('ra')
    >>> dec = data.find_component_id('dec')
    >>> av = data.find_component_id('av')
    >>> alpha = data.find_component_id('alpha')

To find the intersection of the two subsets we have already defined
(i.e., the high-av sources also in the cluster NGC1333)::

   >>> new_subset = high_av & ngc1333
   >>> new_subset
   <glue.core.subset.Subset at 0x105237990>
   >>> new_subset.label = "av_in_cluster"
   >>> data.add_subset(new_subset)

.. image:: subset_02.png
   :width: 60%

The boolean operators ``&``, ``^``, ``|``, and ``~`` act on subsets to
define new subsets represented by the intersection, exclusive
intersection, union, and inverse, respectively.

You can also build subsets out of inequality constraints on component IDs::

   >>> mid_av = (av > 10) & (av < 30)
   >>> data.add_subset(mid_av)

This selects objects with Av values between 10 and 30 (note that we
have hidden the other subsets, for clarity):

.. image:: subset_03.png
   :width: 60%
