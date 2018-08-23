.. _issue_guide:

Finding issues/projects to work on
==================================

About issues
------------

Glue has an `issue tracker <https://github.com/glue-viz/glue/issues>`_ on GitHub. In GitHub terminology, an *issue* can be:

* A bug report
* A report of installation/usage issues (not all of these end up being bugs)
* A request for a new feature
* A request for improved documentation

Issues can be assigned to specific people, which means that they are planning
on working on it. In principle, you can pick any issue that is not already
assigned! Even in the case where an issue is assigned but there has not been
any recent activity, you can add a comment to ask about whether you can help
with it.

.. note:: If you decide to work on an open issue, please leave a comment on
          it to make sure other people know you are doing so!

Picking an issue
----------------

Picking an issue may seem daunting, but we label issues with various tags to
make it easier for you to find issues that might be interesting to you. Here
are some of the labels (click on the label name to go to those issues):

* `package-novice <https://github.com/glue-viz/glue/labels/package-novice>`_:    these are issues that do not require you to know much about glue before
  starting (there are also matching `package-intermediate <https://github.com/glue-viz/glue/labels/package-intermediate>`_ and `package-expert <https://github.com/glue-viz/glue/labels/package-expert>`_ labels)

* `non-gui <https://github.com/glue-viz/glue/labels/non-gui>`_: these are
  issues that **don't** require any knowledge of how to do GUI programming
  (including Qt). Some of these may of course be hard for other reasons, but
  there are also a number of reasonably straightforward issues that just need
  a little time spent on them.

* `bug <https://github.com/glue-viz/glue/labels/bug>`_: these are bug
  reports. Fixing bugs can sometimes be a nice place to start, because you
  don't have to worry about creating new functionality, just fixing existing
  ones. However, not all bugs are easy, so make sure you check the other
  labels.

* `effort-low <https://github.com/glue-viz/glue/labels/effort-low>`_,
  `effort-medium <https://github.com/glue-viz/glue/labels/effort-medium>`_, and
  `effort-high <https://github.com/glue-viz/glue/labels/effort-high>`_: these
  indicate whether the issue will likely take respectively: at most a few hours
  to fix, at most a few days, and more than a few days. Not all issues are
  labeled with these, because it's not always easy to predict how long issues
  will take to tackle.

* `standalone-project <https://github.com/glue-viz/glue/labels/standalone-project>`_: these are
  issues which are projects rather than simple fixes, in that they could take
  several days or more to implement, and they can sometimes be done in
  several stages. These projects are nicely isolated from the rest of the
  glue development, and have a well defined end goal that will result in a
  shiny new feature in glue. These are great issues to work on if you want to
  get more involved in glue development!

Note that GitHub allows you to filter by multiple labels, so you can for example
search for issues that are both `package-novice and non-gui
<https://github.com/glue-viz/glue/issues?q=is%3Aopen+label%3Apackage-novice+label%3Anon-gui>`_

If you have other ideas of things to implement, you can of course do so, and the
issue tracker is not meant to be an exhaustive list. Just drop us a message on
`glue-viz-dev <https://groups.google.com/forum/#!forum/glue-viz-dev>`_ to let us
know what you are working on!

Getting help
------------

In the event that you need any advice when working on an issue, or get stuck,
just leave a comment on the issue, and one of our friendly developers will
help you out!
