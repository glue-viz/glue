.. _merging:

Merging Datasets
================

If several of your files describe the same items, you can merge them into a
single Glue :class:`~glue.core.data.Data` object. Examples of files that make
sense to merge together include:

 - Two or more images that are pixel-aligned to each other
 - Several catalogs whose rows describe the same objects

Why merge?
----------

For multi-dimensional visualizations (like a scatter plot, or an RGB image),
merging datasets allows you to combine attributes from two different files
into a single visualization. It also guarantees that any subset defined
using attributes from one file can be applied to the entries in another file.

Merging vs Linking
------------------

Merging is a different operation than :ref:`linking <linking>`. The easiest
way to appreciate the difference is to think of spreadsheet-like data.
In Glue, linking two datasets defines a conceptual relationship between
the **columns** of a spreadsheet (e.g., two spreadsheets have a column
called "age", but row N describes a different object in each spreadsheet).

Merging, on the other hand, indicates that two spreadsheets are pre-aligned
along each **row** (e.g. row N describes the same item in every spreadsheet, but
the columns of each spreadsheet might be different).

Merging collapses several datasets into a single dataset, while
linking keeps each dataset separate.

How to merge datasets
---------------------

You can merge datasets by highlighting the relevant datasets in the left panel,
right-clicking, and selecting **Merge datasets**.

To merge datasets programmatically, use the :meth:`DataCollection.merge
<glue.core.data_collection.DataCollection.merge>` method.

.. note::

    Datasets should only be merged if each element describes the same item
    in each file. Consequently, all merged datasets must have the same
    number of elements.
