Performance Evaluation
========================

This package provides tools to assess the performance of a machine learning algorithm.

Classification Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: correctrate(gt, pred)

    Compute correct rate of predictions given by ``pred`` w.r.t. the ground truths given in ``gt``.

.. function:: errorrate(gt, pred)

    Compute error rate of predictions given by ``pred`` w.r.t. the ground truths given in ``gt``.

.. function:: confusmat(gt, pred)

    Compute the confusion matrix of the predictions given by ``pred`` w.r.t. the ground truths given in ``gt``.

    It returns an integer matrix ``R`` of size ``(k, k)`` where k is the number of classes in ``gt``, 
    such that ``R(i, j) == countnz((gt .== i) & (pred .== j))``.

    **Examples:**

    .. code-block:: julia

        julia> gt = [1, 1, 1, 2, 2, 2, 3, 3];

        julia> pred = [1, 1, 2, 2, 2, 3, 3, 3];

        julia> C = confusmat(gt, pred)   # compute confusion matrix
        3x3 Array{Int64,2}:
         2  1  0
         0  2  1
         0  0  2

        julia> C ./ sum(C, 2)   # normalize per class 
        3x3 Array{Float64,2}:
         0.666667  0.333333  0.0     
         0.0       0.666667  0.333333
         0.0       0.0       1.0

        julia> trace(C) / length(gt)  # compute correct rate from confusion matrix
        0.75

        julia> correctrate(gt, pred)
        0.75

Hit rate (for retrieval tasks)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: hitrate(gt, ranklist, k)

    Compute the hitrate of rank ``k`` for a ranked list of predictions given by ``ranklist`` w.r.t. the ground truths given in ``gt``. 

    Particularly, if ``gt[i]`` is contained in ``ranklist[1:k, i]``, then the prediction for the ``i``-th sample is said to be *hit within rank ``k``*. The hitrate of rank ``k`` is the fraction of predictions that hit within rank ``k``.

.. function:: hitrates(gt, ranklist, ks)

    Compute hit-rates of multiple ranks (as given by a vector ``ks``). It returns a vector of hitrates ``r``, where ``r[i]`` corresponding to the rank ``ks[i]``.

    Note that computing hit-rates for multiple ranks jointly is more efficient than computing them separately.


Receiver Operating Characteristics (ROC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Receiver Operating Characteristics <http://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_ (ROC) is often used to measure the performance of a detector, thresholded classifier, or a verification algorithm.

The ROC Type
--------------

This package uses an immutable type ``ROCNums`` defined below to capture the ROC of an experiment:

.. code-block:: julia

    immutable ROCNums{T<:Real}
        p::T    # positive in ground-truth
        n::T    # negative in ground-truth
        tp::T   # correct positive prediction
        tn::T   # correct negative prediction
        fp::T   # (incorrect) positive prediction when ground-truth is negative
        fn::T   # (incorrect) negative prediction when ground-truth is positive
    end

One can compute a variety of performance measurements from an instance of ``ROCNums`` (say ``r``):

.. function:: true_positive(r)

    the number of true positives (``r.tp``)

.. function:: true_negative(r)

    the number of true negatives (``r.tn``)

.. function:: false_positive(r)

    the number of false positives (``r.fp``)

.. function:: false_negative(r)

    the number of false negatives (``r.fn``)

.. function:: true_postive_rate(r)

    the fraction of positive samples correctly predicted as positive, defined as ``r.tp / r.p``

.. function:: true_negative_rate(r)

    the fraction of negative samples correctly predicted as negative, defined as ``r.tn / r.n``

.. function:: false_positive_rate(r)
    
    the fraction of negative samples incorrectly predicted as positive, defined as ``r.fp / r.n``

.. function:: false_negative_rate(r)

    the fraction of positive samples incorrectly predicted as negative, defined as ``r.fn / r.p``

.. function:: recall(r)

    Equivalent to ``true_positive_rate(r)``.

.. function:: precision(r)

    the fraction of positive predictions that are correct, defined as ``r.tp / (r.tp + r.fp)``.

.. function:: f1score(r)

    the harmonic mean of ``recall(r)`` and ``precision(r)``.

Computing ROC Curves
---------------------

The package provides a function ``roc`` to compute an instance of ``ROCNums`` or a sequence of such instances from predictions.

.. function:: roc(gt, pred)

    Compute an ROC instance based on ground-truths given in ``gt`` and predictions given in ``pred``.

.. function:: roc(gt, scores, thres[, ord])

    Compute an ROC instance or an ROC curve (a vector of ``ROC`` instances), based on given scores and a threshold ``thres``. 

    Prediction will be made as follows:

    - When ``ord = Forward``: predicts ``1`` when ``scores[i] >= thres`` otherwise 0.
    - When ``ord = Reverse``: predicts ``1`` when ``scores[i] <= thres`` otherwise 0.

    When ``ord`` is omitted, it is defaulted to ``Forward``.

    **Returns:**

    - When ``thres`` is a single number, it produces a single ``ROCNums`` instance; 
    - When ``thres`` is a vector, it produces a vector of ``ROCNums`` instances. 

    **Note:** Jointly evaluating an ROC curve for multiple thresholds is generally much faster than evaluating for them individually.


.. function:: roc(gt, (preds, scores), thres[, ord])

    Compute an ROC instance or an ROC curve (a vector of ``ROC`` instances) for multi-class classification, based on given predictions, scores and a threshold ``thres``. 

    Prediction is made as follows:

    - When ``ord = Forward``: predicts ``preds[i]`` when ``scores[i] >= thres`` otherwise 0.
    - When ``ord = Reverse``: predicts ``preds[i]`` when ``scores[i] <= thres`` otherwise 0.

    When ``ord`` is omitted, it is defaulted to ``Forward``.

    **Returns:**

    - When ``thres`` is a single number, it produces a single ``ROCNums`` instance.
    - When ``thres`` is a vector, it produces an ROC curve (a vector of ``ROCNums`` instances). 

    **Note:** Jointly evaluating an ROC curve for multiple thresholds is generally much faster than evaluating for them individually.

.. function:: roc(gt, scores, n[, ord])

    Compute an ROC curve (a vector of ``ROC`` instances), with respect to ``n`` evenly spaced thresholds from ``minimum(scores)`` and ``maximum(scores)``. (See above for details)

.. function:: roc(gt, (preds, scores), n[, ord])

    Compute an ROC curve (a vector of ``ROC`` instances) for multi-class classification, with respect to ``n`` evenly spaced thresholds from ``minimum(scores)`` and ``maximum(scores)``. (See above for details)

.. function:: roc(gt, scores, ord])

    Equivalent to ``roc(gt, scores, 100, ord)``.

.. function:: roc(gt, (preds, scores), ord])

    Equivalent to ``roc(gt, (preds, scores), 100, ord)``.

.. function:: roc(gt, scores)

    Equivalent to ``roc(gt, scores, 100, Forward)``.

.. function:: roc(gt, (preds, scores))

    Equivalent to ``roc(gt, (preds, scores), 100, Forward)``.


