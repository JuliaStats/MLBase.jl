Cross Validation
==================

This package implements several cross validation schemes: ``Kfold``, ``LOOCV``, and ``RandomSub``. Each scheme is an iterable object, of which each element is a vector of indices (indices of samples selected for training).

Cross Validation Schemes
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: Kfold(n, k)

    ``k``-fold cross validation over a set of ``n`` samples, which are randomly partitioned into ``k`` disjoint validation sets of nearly the same sizes. This generates ``k`` training subsets of length about ``n*(1-1/k)``.

    .. code-block:: julia

        julia> collect(Kfold(10, 3))
        3-element Array{Any,1}:
         [1,3,4,6,7,8,10]
         [2,5,7,8,9,10]
         [1,2,3,4,5,6,9]
    
.. function:: StratifiedKfold(strata, k)

    Like ``Kfold``, but indexes in each strata (defined by unique values of an iterator `strata`) are distributed approximately equally across the ``k`` folds. Each strata should have at least ``k`` members.

    .. code-block:: julia

        julia> collect(StratifiedKfold([:a, :a, :a, :b, :b, :c, :c, :a, :b, :c], 3))
        3-element Array{Any,1}:
         [1,2,4,6,8,9,10]
         [3,4,5,7,8,10]
         [1,2,3,5,6,7,9]
    

.. function:: LOOCV(n)

    Leave-one-out cross validation over a set of ``n`` samples.

    .. code-block:: julia

        julia> collect(LOOCV(4))
        4-element Array{Any,1}:
         [2,3,4]
         [1,3,4]
         [1,2,4]
         [1,2,3]
    

.. function:: RandomSub(n, sn, k)

    Repetitively random subsampling. Particularly, this generates ``k`` subsets of length ``sn`` from a data set with ``n`` samples. 

    .. code-block:: julia

        julia> collect(RandomSub(10, 5, 3))
        3-element Array{Any,1}:
         [1,2,5,8,9] 
         [2,5,7,8,10]
         [1,3,5,6,7] 
    
.. function:: StratifiedRandomSub(strata, sn, k)

    Like ``RandomSub``,  but indexes in each strata (defined by unique values of an iterator `strata`) are distributed approximately equally across the ``k`` subsets.
    ``sn`` should be greater than the number of strata, so that each stratum can be represented in each subset.

    .. code-block:: julia

        julia> collect(StratifiedRandomSub([:a, :a, :a, :b, :b, :c, :c, :a, :b, :c], 7, 5))
        5-element Array{Any,1}:
         [1,2,3,4,6,7,9]
         [1,3,4,6,8,9,10]
         [1,3,5,7,8,9,10]
         [1,2,4,7,8,9,10]
         [1,2,3,4,5,6,10]
    

Cross Validation Function
~~~~~~~~~~~~~~~~~~~~~~~~~~

The package also provides a function ``cross_validate`` as below to run a cross validation procedure.

.. function:: cross_validate(estfun, evalfun, n, gen)

    Run a cross validation procedure.

    :param estfun: The estimation function, which takes a vector of training indices as input and returns a learned model, as:

        .. code-block:: julia

            model = estfun(train_inds)
        

    :param evalfun: The evaluation function, which takes a model and a vector of testing indices as input and returns a score that indicates the goodness of the model, as

        .. code-block:: julia

            score = evalfun(model, test_inds)

    :param n: The total number of samples.

    :param gen: An iterable object that provides training indices, *e.g.*, one of the cross validation schemes listed above.

    :return: a vector of scores obtained in the multiple runs.

    **Example:**

    .. code-block:: julia

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
        scores = cross_validate(
            inds -> compute_center(data[:, inds]),        # training function
            (c, inds) -> compute_rmse(c, data[:, inds]),  # evaluation function
            n,              # total number of samples
            Kfold(n, 5))    # cross validation plan: 5-fold 

        # get the mean and std of the scores
        (m, s) = mean_and_std(scores)

    
    Please refer to ``examples/crossval.jl`` for the entire script.

