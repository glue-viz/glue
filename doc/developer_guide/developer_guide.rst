Developer Guide
===============

So you want to help develop Glue? Let's get started!

First, be sure to join the `glueviz-dev
<https://groups.google.com/forum/#!forum/glue-viz-dev>`_ mailing list for any
questions or discussions related to development, and let us know if any of the
documentation below is unclear. You are also very welcome to introduce yourself on the list and let us know that you are interested in contributing!

If you want to contribute, but don't yet have a specific idea of where to make contributions, you can check out our `Issue tracker <https://github.com/glue-viz/glue/issues>`_. We use different labels to categorize the issues, and one label you might be interested in is `package-novice <https://github.com/glue-viz/glue/issues?q=is%3Aissue+is%3Aopen+label%3APackage-novice>`_, which highlights issues that don't require an in-depth understanding of the Glue package, but only require you to understand/edit a small part of the code base (which is typically mentioned in the issue).

Architecture
------------

The pages below take you through the main infrastructure in Glue, and in
particular how selections, linking, and communications are handled internally.
You don't need to understand all of this in order to get started with
contributing, but in order to tackle some of the more in-depth issues, this
will become important. This is not meant to be a completely exhaustive guide,
but if there are areas that you feel could be explained better, or are missing
and would be useful, please let us know!

.. toctree::
   :maxdepth: 1

   organization.rst
   data.rst
   selection.rst
   communication.rst
   linking.rst
   
Developer logistics
-------------------

The following pages provide more logistical information about contributing to Glue, including how to actually make the code contributions, and how the testing framework is set up.
  
.. toctree::
   :maxdepth: 1
  
   coding_guidelines.rst
   testing.rst
   app_building.rst
   api.rst