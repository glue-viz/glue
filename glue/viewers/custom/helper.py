# This is a helper function that gets imported into the top-level namespace of glue.

__all__ = ['custom_viewer']


def custom_viewer(name, **kwargs):
    """
    Create a custom interactive data viewer.

    To use this, first create a new variable by calling custom_viewer.
    Then, register one or more viewer functions using decorators.

    :param name: The name of the new viewer
    :type name: str

    Named arguments are used to build widgets and pass data
    to viewer functions. See ``specifying widgets`` below.

    Example::

      v = custom_viewer('My custom viewer', check=False, x='att(x)')

      @v.setup
      def setup_func(axes):
        ''' Setup the plot when the viewer is created '''
        ...

      @v.plot_data
      def plot_data_func(axes, check, style):
          ''' Visualize a full dataset '''
          ...

      @v.plot_subset
      def plot_subset_func(axes, check, style):
          ''' Visualize a subset '''
          ...

      @v.update_settings
      def update_settings_func(check):
          ''' Respond to the user changing a widget setting '''
          ...

      @v.select
      def select(roi, x):
        ''' Filter a dataset based on an roi. Return a boolean array '''
        ...


      @v.make_selector
      def make_selector_func(roi):
          ''' Turn a roi into a subset state '''
          ...


    **Specifying Widgets**

    Keywords passed to ``custom_viewer`` serve two purposes: they
    setup information to be passed into the viewer functions, and
    they create widgets. The type of widget that is created depends
    on the keyword value:

      * ``keyword=False | True`` creates a checkbox. The check state
        is passed as a Boolean into the viewer functions
      * ``keyword=(10, 20, [15])`` creates a slider. The current value
        of the slider is passed as a number to the viewer functions.
        The first two numbers specify the minimum and maximum allowed value,
        while the optional third number specifies the initial value.
      * ``keyword=['a', 'b', 'c']`` creates a dropdown menu. The current
        selection is passed as a string to the viewer functions.
      * ``keyword={'a':1, 'b':2}`` behaves similarly to the lists above,
        but uses the keys as dropdown labels and values as the setting
        passed to viewer functions.
      * ``keyword='att(foo)'`` doesn't create any widget, but passes
        in the attribute named ``foo`` to the viewer functions, as an
        :class:`~glue.viewers.custom.qt.custom_viewer.AttributeWithInfo` object.
      * ``keyword='att'`` creates a dropdown to let the user select
        one of the attributes from the data. The selected attribute
        is passed as an :class:`~glue.viewers.custom.qt.custom_viewer.AttributeWithInfo`

    **Viewer Functions**

    Custom viewers can implement any of the following functions:

     * ``setup_func`` is called once, when the viewer is created.
     * ``plot_data`` is called to update the visualization of a
       full dataset.
     * ``plot_subset`` is used to visualize data subsets.
     * ``update_settings`` is called whenever a user modifies
       a widget setting.
     * ``select`` specifies how user-drawn regions on the viewer
       are used to filter data. It has access to an :class:`~glue.core.roi.Roi`
       input, and returns a Boolean array testing whether each element
       in a dataset is part of a subset.
     * ``make_selector`` is an alternative to ``select``. Instead of returning
        an array, ``make_selector`` returns a :class:`~glue.core.subset.SubsetState`
    """
    # For now we only support Qt, but this function should be extended to work
    # with non-Qt front-ends in future.
    from glue.viewers.custom.qt import CustomViewer
    return CustomViewer.create_new_subclass(name, **kwargs)
