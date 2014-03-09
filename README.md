# MLBase.jl

[![Build Status](https://travis-ci.org/JuliaStats/MLBase.jl.png)](https://travis-ci.org/JuliaStats/MLBase.jl)

Swiss knife for machine learning. 

This package does not implement specific machine learning algorithms. Instead, it provides a collection of useful tools to support machine learning programs, including:

- Data manipulation
- Standardization
- Score-based classification
- Cross validation
- Performance evaluation (e.g. evaluating ROC)
- Computing deviation between arrays

-----------

## Data Manipulation

- **repeach**(a, n)

    Repeat each element in vector ``a`` for ``n`` times. Here ``n`` can be either a scalar or a vector with the same length as ``a``.

    ```julia
    using MLBase

    repeach(1:3, 2) # --> [1, 1, 2, 2, 3, 3]
    repeach(1:3, [3,2,1]) # --> [1, 1, 1, 2, 2, 3]
    ```

- **repeachcol**(a, n)

    Repeat each column in matrix ``a`` for ``n`` times. Here ``n`` can be either a scalar or a vector with ``length(n) == size(a,2)``.

- **repeachrow**(a, n)

    Repeat each row in matrix ``a`` for ``n`` times. Here ``n`` can be either a scalar or
    a vector with ``length(n) == size(a,1)``.

- **counteq**(a, b) 

    Count the number of occurences of ``a[i] == b[i]``.

- **countne**(a, b)

    Count the number of occurrences of ``a[i] != b[i]``.    


## Data Standardization

Sometimes, it might be desirable to standardize a set of data before feeding it to a machine learning task (*e.g.* PCA), in order to balance the contributions of different components. 

The package provides a ``Standardize`` type to capture the standardization transform, which is defined as below:

```julia
immutable Standardize
    dim::Int
    mean::Vector{Float64}
    scale::Vector{Float64}
end
```
Applying a standardization transform ``t`` to a vector ``x`` is defined as:

```julia
y[i] = t.scale[i] * (x[i] - t.mean[i])
```
Here, ``t.scale[i]`` is the inverse of the standard deviation of the i-th variable. After standarization, each component would have zero mean and unit standard deviation.

Note we allow either ``mean`` or ``scale`` fields to be empty, which indicates that the step of shifting the mean or that of scaling the component would not be applied. 

- **estimate**(Standardize, X[; center=true, scale=true])

    Estimate a standardization transform from a given data set ``X``. 

    This package follows the convention that each column of ``X`` is an observation and each row is a component/variable. 

- **standardize**(X[; center=true, scale=true])

    Estimate a standardization transform from ``X`` and apply it to ``X``. It returns a pair ``(Y, t)``, where ``Y`` is the transformed data matrix, and ``t`` is an instance of ``Standardize`` that represents the estimated transform.

- **standardize!**(X[; center=true, scale=true])

    Similar to ``standardize``, except that the transformation to ``X`` happens inplace.

- **transform**(t, X)

    Apply a standardization transform ``t`` to ``X``, return the transformed vector/matrix.

- **transform!**(t, X)

    Apply a standardization transform ``t`` to ``X`` inplace, return ``X``.


## Label Map

In machine learning, we often need to first attach each class with an integer label. This package provides a type ``LabelMap`` that captures the association between discrete values (*e.g* a finite set of strings) and integer labels. 

Together with ``LabelMap``, the package also provides a function ``labelmap`` to construct the map from a sequence of discrete values, and a function ``labelencode`` to map discrete values to integer labels. 

```julia
julia> lm = labelmap(["a", "a", "b", "b", "c"])
LabelMap (with 3 labels):
[1] a
[2] b
[3] c

julia> labelencode(lm, "b")
2

julia> labelencode(lm, ["a", "c", "b"])
3-element Array{Int64,1}:
 1
 3
 2
```
Note that ``labelencode`` can be applied to either single value or an array.

The package also provides a function ``groupindices`` to group indices based on associated labels. 

```julia
julia> groupindices(3, [1, 1, 1, 2, 2, 3, 2])
3-element Array{Array{Int64,1},1}:
 [1,2,3]
 [4,5,7]
 [6]    

 # using lm as constructed above
julia> groupindices(lm, ["a", "a", "c", "b", "b"])
3-element Array{Array{Int64,1},1}:
 [1,2]
 [4,5]
 [3]
```

## Score-based Classification

No matter how sophisticated a classification framework is, the entire classification task generally consists of two steps: (1) assign a score/distance to each class, and (2) choose the class that yields the highest score/lowest distance.

This package provides a function ``classify`` and its friends to accomplish the second step.

- **classify**(x[, ord])

    Classify based on scores given in ``x`` and the order of scores specified in ``ord``.

    Generally, ``ord`` can be any instance of type ``Ordering``. However, it usually enough to use either ``Forward`` or ``Reverse``:

    - ``ord = Forward``: higher value indicates better match (*e.g.*, similarity)
    - ``ord = Reverse``: lower value indicates better match (*e.g.*, distances)

    When ``ord`` is omitted, it is defaulted to ``Forward``.

    When ``x`` is a vector, it produces an integer label. When ``x`` is a matrix, it produces a vector of integers, each for a column of ``x``.

    ```julia
    classify([0.2, 0.5, 0.3])  # --> 2
    classify([0.2, 0.5, 0.3], Forward)  # --> 2
    classify([0.2, 0.5, 0.3], Reverse)  # --> 1

    classify([0.2 0.5 0.3; 0.7 0.6 0.2]') # --> [2, 1]
    classify([0.2 0.5 0.3; 0.7 0.6 0.2]', Forward) # --> [2, 1]
    classify([0.2 0.5 0.3; 0.7 0.6 0.2]', Reverse) # --> [1, 3]
    ```

- **classify!**(r, x[, ord])

    Write predicted labels to ``r``. 

- **classify_withscore**(x[, ord])

    Return a pair as ``(label, score)``, where ``score`` is the input score corresponding to the predicted label.

- **classify_withscores**(x[, ord])

    This function applies to a matrix ``x`` comprised of multiple samples (each being a column). It returns a pair ``(labels, scores)``.

- **classify_withscores!**(r, s, x[, ord])

    Write predicted labels to ``r`` and corresponding scores to ``s``.


## Cross Validation

This package implements several cross validation schemes: ``Kfold``, ``LOOCV``, and ``RandomSub``. Each scheme is an iterable object, of which each element is a vector of indices (indices of samples selected for training).

- **Kfold**(n, k)

    ``k``-fold cross validation over a set of ``n`` samples, which are randomly partitioned into ``k`` disjoint subsets of nearly the same sizes.

    ```julia
    julia> collect(Kfold(10, 3))
    3-element Array{Any,1}:
     [1,2,7]  
     [4,5,8,9]
     [3,6,10]
    ```

- **LOOCV**(n)

    Leave-one-out cross validation over a set of ``n`` samples.

    ```julia
    julia> collect(LOOCV(4))
    4-element Array{Any,1}:
     [2,3,4]
     [1,3,4]
     [1,2,4]
     [1,2,3]
    ```

- **RandomSub**(n, sn, k)

    Repetitively random subsampling. Particularly, this generates ``k`` subsets of length ``sn`` from a data set with ``n`` samples. 

    ```julia
    julia> collect(RandomSub(10, 5, 3))
    3-element Array{Any,1}:
     [1,2,5,8,9] 
     [2,5,7,8,10]
     [1,3,5,6,7] 
    ``` 

The package also provides a function ``cross_validate`` as below to run a cross validation procedure.

- **cross_validate**(estfun, evalfun, n, gen, ord)

    Run a cross validation procedure.

    - ``estfun``:  estimation function, which takes a vector of training indices as input and returns a learned model, as

        ```julia
        model = estfun(train_inds)
        ```

    - ``evalfun``: evaluation function, which takes a model and a vector of testing indices as input and returns a score that indicates the goodness of the model, as

        ```julia
        score = evalfun(model, test_inds)
        ```

    - ``n``: the total number of samples

    - ``gen``: an iterable object that provides training indices, *e.g.*, a cross validation scheme as listed above.

    - ``ord``: the ordering of the evaluated score. ``ord = Forward`` means that higher score indicates better model; ``ord = Reverse`` means that lower score indicates better model.

    This function returns a tuple as ``(best_model, best_score, best_indices)``.

    Here is a full example:

    ```julia
    # A simple example to demonstrate the use of cross validation
    #
    # Here, we consider a simple model: using a mean vector to represent
    # a set of samples. The goodness of the model is assessed in terms
    # of the RMSE (root-mean-square-error) evaluated on the testing set
    #

    using MLBase

    # functions
    compute_center(X::Matrix{Float64}) = vec(mean(X, 2))

    compute_rmse(c::Vector{Float64}, X::Matrix{Float64}) = 
        sqrt(mean(sum(abs2(X .- c),1)))

    # data
    const n = 200
    const data = [2., 3.] .+ randn(2, n)

    # cross validation
    (c, v, inds) = cross_validate(
        inds -> compute_center(data[:, inds]),        # training function
        (c, inds) -> compute_rmse(c, data[:, inds]),  # evaluation function
        n,              # total number of samples
        Kfold(n, 5),    # cross validation plan: 5-fold 
        Reverse)        # smaller score indicates better model
    ```

    Please refer to ``examples/crossval.jl`` for the entire script.


## Performance Evaluation

This package provides tools to assess the performance of a machine learning algorithm.

#### Correct rate and error rate

- **correctrate**(gt, pred)

    Compute correct rate of predictions given by ``pred`` w.r.t. the ground truths given in ``gt``.

- **errorrate**(gt, pred)

    Compute error rate of predictions given by ``pred`` w.r.t. the ground truths given in ``gt``.

#### Hit rate

- **hitrate**(gt, ranklist, k)

    Compute the hitrate of rank ``k`` for a ranked list of predictions given by ``ranklist`` w.r.t. the ground truths given in ``gt``. 

    Particularly, if ``gt[i]`` is contained in ``ranklist[1:k, i]``, then the prediction for the ``i``-th sample is said to be *hit within rank ``k``*. The hitrate of rank ``k`` is the fraction of predictions that hit within rank ``k``.

- **hitrates**(gt, ranklist, ks)

    Compute hit-rates of multiple ranks (as given by a vector ``ks``). It returns a vector of hitrates ``r``, where ``r[i]`` corresponding to the rank ``ks[i]``.

    Note that computing hit-rates for multiple ranks jointly is more efficient than computing them separately.

#### ROC

ROC (Receiver Operating Characteristics) is often used to measure the performance of a detector, thresholded classifier, or a verification algorithm.

This package uses an immutable type ``ROCNums`` defined below to capture the ROC of an experiment:

```julia
immutable ROCNums{T<:Real}
    p::T    # positive in ground-truth
    n::T    # negative in ground-truth
    tp::T   # correct positive prediction
    tn::T   # correct negative prediction
    fp::T   # (incorrect) positive prediction when ground-truth is negative
    fn::T   # (incorrect) negative prediction when ground-truth is positive
end
```

One can compute a variety of performance measurements from an instance of ``ROCNums`` (say ``r``):

- **true_positive**(r)

    the number of true positives (``r.tp``)

- **true_negative**(r)

    the number of true negatives (``r.tn``)

- **false_positive**(r)

    the number of false positives (``r.fp``)

- **false_negative**(r)

    the number of false negatives (``r.fn``)

- **true_postive_rate**(r)

    the fraction of positive samples correctly predicted as positive, defined as ``r.tp / r.p``

- **true_negative_rate**(r)

    the fraction of negative samples correctly predicted as negative, defined as ``r.tn / r.n``

- **false_positive_rate**(r)
    
    the fraction of negative samples incorrectly predicted as positive, defined as ``r.fp / r.n``

- **false_negative_rate**(r)

    the fraction of positive samples incorrectly predicted as negative, defined as ``r.fn / r.p``

- **recall**(r)

    Equivalent to ``true_positive_rate(r)``.

- **precision**(r)

    the fraction of positive predictions that are correct, defined as ``r.tp / (r.tp + r.fp)``.

- **f1score**(r)

    the harmonic mean of ``recall(r)`` and ``precision(r)``.


The package provides a function ``roc`` to compute an instance of ``ROCNums`` or a sequence of such instances from predictions.

- **roc**(gt, pred)

    compute an ROC instance based on ground-truths given in ``gt`` and predictions given in ``pred``.

- **roc**(gt, scores, thres[, ord])

    compute an ROC instance based on scores and a threshold ``thres``. 

    Prediction is made as follows:
    - when ``ord = Forward``: predicts 1 when ``scores[i] >= thres`` otherwise 0.
    - when ``ord = Reverse``: predicts 1 when ``scores[i] <= thres`` otherwise 0.

    When ``ord`` is omitted, it is defaulted to ``Forward``.

    When ``thres`` is a single number, it produces a single ``ROCNums`` instance; when ``thres`` is a vector, it produces a vector of ``ROCNums`` instances. Jointly evaluating the ROC for multiple thresholds is generally much faster than evaluating for them individually.


- **roc**(gt, (preds, scores), thres[, ord])

    compute an ROC instance based on (unthresholded) predictions, scores and a threshold ``thres``. 

    Prediction is made as follows:
    - when ``ord = Forward``: predicts ``preds[i]`` when ``scores[i] >= thres`` otherwise 0.
    - when ``ord = Reverse``: predicts ``preds[i]`` when ``scores[i] <= thres`` otherwise 0.

    When ``ord`` is omitted, it is defaulted to ``Forward``.

    When ``thres`` is a single number, it produces a single ``ROCNums`` instance; when ``thres`` is a vector, it produces a vector of ``ROCNums`` instances. Jointly evaluating the ROC for multiple thresholds is generally much faster than evaluating for them individually.

- **roc**(gt, scores, n[, ord])
- **roc**(gt, (preds, scores), n[, ord])

    compute a sequence of ROC instances for ``n`` evenly spaced thresholds from ``minimum(scores)`` and ``maximum(scores)``.

- **roc**(gt, scores, ord])
- **roc**(gt, (preds, scores), ord])

    Respectively equivalent to ``roc(gt, scores, 100, ord)`` and ``roc(gt, (preds, scores), 100, ord)``.

- **roc**(gt, scores)
- **roc**(gt, (preds, scores))

    Respectively equivalent to ``roc(gt, scores, 100, Forward)`` and ``roc(gt, (preds, scores), 100, Forward)``.


## Computing Deviations

For many machine learning algorithms, the primary goal is to minimize the deviation between the ground-truth signals and the reconstructed ones. This package provides functions to compute various deviations:

- **sqL2dist**(a, b)

    Squared L2 distance between ``a`` and ``b``.

- **L2dist**(a, b)

    L2 distance between ``a`` and ``b``.

- **L1dist**(a, b)

    L1 distance between ``a`` and ``b``.

- **Linfdist**(a, b)

    Linf distance between ``a`` and ``b``.

- **gkldiv**(a, b)

    Generalized Kullback-Leibler divergence between two arrays ``a`` and ``b``.

    Defined as ``sum(a .* log(a / b) - a + b)``. Note: when ``sum(a) == 1`` and ``sum(b) == 1``, it reduces to the KL-divergence in standard sense.

- **meanad**(a, b)

    Mean absolute deviation between ``a`` and ``b``, *i.e.* ``mean(abs(a - b))``.

- **maxad**(a, b)

    Maximum absolute deviation between ``a`` and ``b``, *i.e.* ``maximum(abs(a - b))``.

- **msd**(a, b)

    Mean squared deviation between ``a`` and ``b``, *i.e.* ``mean(abs2(a - b))``.

- **rmsd**(a, b)

    Root mean squared deviation, *i.e.* ``sqrt(msd(a, b))``.

- **nrmsd**(a, b)

    Normalized root mean squared deviation, *i.e.* ``rmsd(a, b) / (maximum(a) - minimum(a))``.

- **psnr**(a, b, maxv)

    Peak signal-to-noise ratio, *i.e.* ``10 * log10(maxv^2 / msd(a, b))``.

**Note:** all these functions are implemented in a reasonably efficient way without creating any temporary arrays in the middle.

