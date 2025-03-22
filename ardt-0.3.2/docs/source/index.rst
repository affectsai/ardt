.. Affective Research Dataset Toolkit documentation master file, created by
   sphinx-quickstart on Wed Mar 19 13:07:36 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Affective Research Dataset Toolkit documentation
================================================

Section
-------

term (up to a line of text)
  Definition of the term, which must be indented

  and can even consist of multiple paragraphs

next term
  Description.

| These lines are
| broken exactly like in
| the source file.

* This is a *bulleted* list.
* It has two items, the second
  item uses two lines.

#. This is a **numbered** list.
#. It has two items too.

.. code-block:: html
    :linenos:

    <h1>code block example</h1>


This is a normal text paragraph. The next paragraph is a code sample::

  import math
  print("import done")

This is a normal text paragraph again.


This is a paragraph that contains `a link`_.

.. _a link: https://www.example.com/

=====  =====  =======
A      B      A and B
=====  =====  =======
False  False  False
True   False  False
=====  =====  =======

.. literalinclude:: ../../README.md

.. toctree::
    :hidden:
    :titlesonly:

    user_guide/index
    reference/index
    changelog