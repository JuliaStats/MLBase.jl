## MLBase.jl

[![Build Status](https://travis-ci.org/JuliaStats/MLBase.jl.png)](https://travis-ci.org/JuliaStats/MLBase.jl)

Swiss knife for machine learning. Particularly, it provides a collection of useful tools for machine learning programs, including:

- Data manipulation
- Score-based classification
- Cross validation
- Performance evaluation (e.g. evaluating ROC)

-----------

### Data Manipulation

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


### Label Map

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

### Score-based Classification

No matter how sophisticated a classification framework is, the entire classification task generally consists of two steps: (1) assign a score/distance to each class, and (2) choose the class that yields the highest score/lowest distance.

This package provides a function ``classify`` and its friends to accomplish the second step.

- **classify**(x, ord)

    Classify based on scores given in ``x`` and the order of scores specified in ``ord``.

    Generally, ``ord`` can be any instance of type ``Ordering``. However, it usually enough to use either ``Forward`` or ``Reverse``:

    - ``ord = Forward``: higher value indicates better match (*e.g.*, similarity)
    - ``ord = Reverse``: lower value indicates better match (*e.g.*, distances)

    When ``x`` is a vector, it produces an integer label. When ``x`` is a matrix, it produces a vector of integers, each for a column of ``x``.

    ```julia
    classify([0.2, 0.5, 0.3], Forward)  # --> 2
    classify([0.2, 0.5, 0.3], Reverse)  # --> 1

    classify([0.2 0.5 0.3; 0.7 0.6 0.2]', Forward) # --> [2, 1]
    classify([0.2 0.5 0.3; 0.7 0.6 0.2]', Reverse) # --> [1, 3]
    ```

- **classify**(x)

    Equivalent to ``classify(x, Forward)``.

- **classify!**(r, x, ord)

    Write predicted labels to ``r``. 

- **classify!**(r, x)

    Equivalent to ``classify!(r, x, Forward)``.

- **classify_withscore**(x, ord)

    Return a pair as ``(label, score)``, where ``score`` is the input score corresponding to the predicted label.

- **classify_withscore**(x)

    Equivalent to ``classify_withscore(x, Forward)``.

- **classify_withscores**(x, ord)

    This function applies to a matrix ``x`` comprised of multiple samples (each being a column). It returns a pair ``(labels, scores)``.

- **classify_withscores**(x)

    Equivalent to ``classify_withscores(x, Forward)``.

- **classify_withscores!**(r, s, x, ord)

    Write predicted labels to ``r`` and corresponding scores to ``s``.

- **classify_withscores!**(r, s, x)

    Equivalent to ``classify_withscores!(r, s, x, Forward)``.


### Cross Validation

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









