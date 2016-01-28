Building Custom Data Viewers
============================

.. figure:: images/bball_3.png
   :align: center

Glue's standard data viewers (scatter plots, images, histograms) are
useful in a wide variety of data exploration settings. However, they
represent a *tiny* fraction of the ways to view a particular
dataset. For this reason, Glue provides a simple mechanism
for creating custom visualizations using matplotlib.

Creating a :func:`custom data viewer <glue.custom_viewer>` requires writing a little bit of Matplotlib
code but involves little to no GUI programming. The next several
sections illustrate how to build a custom data viewer by example.


The Goal: Basketball Shot Charts
--------------------------------
In Basketball, Shot Charts show the spatial distribution of shots
for a particiular player, team, or game. The `New York Times <http://www.nytimes.com/interactive/2012/06/11/sports/basketball/nba-shot-analysis.html?_r=0>`_ has a nice example.

There are three basic features that we might want to incorporate into
a shot chart:

 * The distribution of shots (or some statistic like the success rate), shown as a heatmap in the background.
 * The locations of a particular subset of shots, perhaps plotted as
   points in the foreground
 * The relevant court markings, like the 3-point line and hoop location.

We'll build a Shot Chart in Glue incrementally, starting with the simplest
code that runs.


Shot Chart Version 1: Heatmap and plot
--------------------------------------

Our first attempt at a shot chart will draw the heatmap of all shots,
and overplot shot subsets as points. Here's the code:

.. literalinclude:: scripts/bball_viewer_1.py
   :linenos:

Before looking at the code itself, let's look at how it's used. If you
include or import this code in your :ref:`config.py <configuration>` file, Glue will recognize the new viewer.
Open `this shot catalog <https://raw.githubusercontent.com/ChrisBeaumont/basketball/master/shots.csv>`_, and create a new
shot chart with it. You'll get something that looks like this:

.. figure:: images/bball_1.png
   :align: center

Furthermore, subsets that we define (e.g., by selecting regions of a
histogram) are shown as points (notice that Tim Duncan's shots are concentrated closer to the hoop).

.. figure:: images/bball_2.png
   :align: center

Let's look at what the code does. Line 5 creates a new custom viewer,
and gives it the name ``Shot Plot``. It also specifies ``x`` and ``y`` keywords which we'll come back to shortly (spoiler: they tell Glue to
pass data attributes named ``x`` and ``y`` to ``show_hexbin``).

Line 11 defines a ``show_hexbin`` function, that visualizes a dataset
as a heatmap. Furthermore, the decorator on line 10 registers this
function as the ``plot_data`` function, responsible for visualizing a dataset as a whole.

Custom functions like ``show_hexbin`` can accept a variety of input
arguments, depending on what they need to do. Glue looks at the names
of the inputs to decide what data to pass along.  In the case of this
function:

 - Arguments named ``axes`` contain the Matplolib Axes object to draw with
 - ``x`` and ``y`` were provided as keywords to ``custom_viewer``. They
   contain the data (as arrays) corresponding to the attributes labeled
   ``x`` and ``y`` in the catalog

The function body itself is pretty simple -- we just use  the
``x`` and ``y`` data to build a hexbin plot in Matplotlib.

Lines 19-25 follow a similar structure to handle the visualization of subsets, by defining a ``plot_subset`` function. We make use of the
``style`` keyword, to make sure we choose colors, sizes, and
opacities that are consistent with the rest of Glue. The value passed
to the style keyword is a :class:`~glue.core.visual.VisualAttributes`
object.

Custom data viewers give you the control to visualize data how you
want, while Glue handles all the tedious bookeeping associated with updating plots when selections, styles, or datasets change. Try it out!

Still, this viewer is pretty limited. In particular, it's missing
court markings, the ability to select data in the plot, and the ability
to interactively change plot settings with widgets. Let's fix that.

Shot Chart Version 2: Court markings
------------------------------------

We'd like to draw court markings to give some context to the heatmap.
This is independent of the data, and we only need to render it once.
Just as you can register data and subset plot functions, you can also
register a setup function that gets called a single time, when the viewer
is created. That's a good place to draw court markings:

.. literalinclude:: scripts/bball_viewer_2.py
   :linenos:

This version adds a new ``draw_court`` function at Line 30. Here's the result:

.. figure:: images/bball_3.png
   :align: center

Shot Chart Version 3: Widgets
-----------------------------
There are several parameters we might want to tweak about our
visualization as we explore the data. For example, maybe we want
to toggle between a heatmap of the shots, and the percentage of
successful shots at each location. Or maybe we want to choose
the bin size interactively.

The keywords that you pass to :func:`~glue.custom_viewer` allow you to
set up this functionality. Keywords serve two purposes: they define
new widgets to interact with the viewer, and they define keywords to pass
onto drawing functions like ``plot_data``.

For example, consider :download:`this version <scripts/bball_viewer_3.py>` of the Shot Plot code:

.. literalinclude:: scripts/bball_viewer_3.py
   :linenos:

This code passes 4 new keywords to :func:`~glue.custom_viewer`:

  * ``bins=(10, 100)`` adds a slider widget, to choose an integer
    between 10 and 100. We'll use this setting to set the
    bin size of the heatmap.
  * ``hitrate=False`` adds a checkbox. We'll use this setting to
    toggle between a heatmap of total shots, and a map of
    shot success rate.
  * ``color=['Reds', 'Purples']`` creates a dropdown list
    of possible colormaps to use for the heatmap.
  * ``hit='att(shot_made)'`` behaves like the x and y keywords from
    earlier -- it doesn't add a new widget, but it will
    pass the shot_made data along to our plotting functions.

This results in the following interface:

.. figure:: images/bball_4.png
   :align: center

Whenever the user changes the settings of these widgets, the
drawing functions are re-called. Furthermore, the current
setting of each widget is available to the plotting functions:

 * ``bins`` is set to an integer
 * ``hitrate`` is set to a boolean
 * ``color`` is set to ``'Reds'`` or ``'Purples'``
 * ``x``, ``y``, and ``hit`` are passed as :class:`~glue.viewers.custom.qt.custom_viewer.AttributeInfo` objects (which are just numpy arrays with a special ``id`` attribute, useful when performing selection below).

The plotting functions can use these variables to draw the appropriate
plots -- in particular, the ``show_hexbin`` function chooses
the binsize, color, and aggregation based on the widget settings.

Shot Chart Version 4: Selection
-------------------------------
One key feature still missing from this Shot Chart is the ability to
select data by drawing on the plot. To do so, we need to write a
``select`` function that computes whether a set of data points are
contained in a user-drawn :class:`region of interest <glue.core.roi.Roi>`:

.. literalinclude:: scripts/bball_viewer_4.py
   :lines: 18-20
   :linenos:

With :download:`this version <scripts/bball_viewer_4.py>` of the code you can how draw shapes on the plot to select data:

.. figure:: images/bball_5.png
   :align: center

Viewer Subclasses
-----------------
The shot chart example used decorators to define custom plot functions.
However, if your used to writing classes you can also subclass
:class:`~glue.viewers.custom.qt.custom_viewer.CustomViewer` directly. The code is largely the
same:

.. literalinclude:: scripts/bball_viewer_class.py
   :linenos:


Valid Function Arguments
------------------------

The following argument names are allowed as inputs to custom
viewer functions:

 - Any UI setting provided as a keyword to :func:`glue.custom_viewer`.
   The value passed to the function will be the current setting of the
   UI element.
 - ``axes`` is the matplotlib Axes object to draw to
 - ``roi`` is the :class:`glue.core.roi.Roi` object a user created --
   it's only available in ``make_selection``.
 - ``style`` is available to ``plot_data`` and ``plot_subset``. It is
   the :class:`~glue.core.visual.VisualAttributes` associated with the
   subset or dataset to draw
 - ``state`` is a general purpose object that you can use to store
   data with, in case you need to keep track of state in between
   function calls.

UI Elements
-----------

Simple user interfaces are created by specifying keywords to
:func:`~glue.custom_viewer` or class-level variables to
:class:`~glue.viewers.custom.qt.custom_viewer.CustomViewer` subclasses. The type of
widget, and the value passed to plot functions, depends on the value
assigned to each variable. See :func:`~glue.custom_viewer` for
information.

Other Guidelines
----------------
 - You can find other example data viewers at `<https://github.com/glue-viz/example_data_viewers>`_. Contributions to this repository are welcome!

 - Glue auto-assigns the z-order of data and subset layers to the values
   [0, N_layers - 1]. If you have elements you want to plot in the
   background, give them a negative z-order

 - Glue tries to keep track of the plot layers that each custom function
   creates, and auto-deletes old layers. This behavior can be disabled
   by setting ``viewer.remove_artists=False``. Likewise,
   ``plot_data`` and ``plot_subset`` can explicitly return a list
   of newly-created artists. This might be more efficient if your
   plot is very complicated.

 - By default, ``plot_data`` and ``plot_subset`` are called whenever
   UI settings change. To disable this behavior, set
   ``viewer.redraw_on_settings_change=False``.

 - By default, Glue sets the margins of figures so that the space between axes
   and the edge of figures is constant in absolute terms. If the default values
   are not adequate for your viewer, you can set the margins in the ``setup``
   method of the custom viewer by doing e.g.::

       axes.resizer.margins = [0.75, 0.25, 0.5, 0.25]

   where the list gives the ``[left, right, bottom, top]`` margins in inches.
