Classification
================

A classification procedure, no matter how sophisticated it is, generally consists of two steps: (1) assign a score/distance to each class, and (2) choose the class that yields the highest score/lowest distance.

This package provides a function ``classify`` and its friends to accomplish the second step, that is, to predict labels based on scores.

.. function:: classify(x[, ord])

    Classify based on scores given in ``x`` and the order of scores specified in ``ord``.

    Generally, ``ord`` can be any instance of type ``Ordering``. However, it usually enough to use either ``Forward`` or ``Reverse``:

    - ``ord = Forward``: higher value indicates better match (*e.g.*, similarity)
    - ``ord = Reverse``: lower value indicates better match (*e.g.*, distances)

    When ``ord`` is omitted, it is defaulted to ``Forward``.

    When ``x`` is a vector, it produces an integer label. When ``x`` is a matrix, it produces a vector of integers, each for a column of ``x``.

    .. code-block:: julia

        classify([0.2, 0.5, 0.3])  # --> 2
        classify([0.2, 0.5, 0.3], Forward)  # --> 2
        classify([0.2, 0.5, 0.3], Reverse)  # --> 1

        classify([0.2 0.5 0.3; 0.7 0.6 0.2]') # --> [2, 1]
        classify([0.2 0.5 0.3; 0.7 0.6 0.2]', Forward) # --> [2, 1]
        classify([0.2 0.5 0.3; 0.7 0.6 0.2]', Reverse) # --> [1, 3]
    

.. function:: classify!(r, x[, ord])

    Write predicted labels to ``r``. 

.. function:: classify_withscore(x[, ord])

    Return a pair as ``(label, score)``, where ``score`` is the input score corresponding to the predicted label.

.. function:: classify_withscores(x[, ord])

    This function applies to a matrix ``x`` comprised of multiple samples (each being a column). It returns a pair ``(labels, scores)``.

.. function:: classify_withscores!(r, s, x[, ord])

    Write predicted labels to ``r`` and corresponding scores to ``s``.
