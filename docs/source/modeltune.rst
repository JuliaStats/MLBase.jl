Model Tuning
================

Many machine learning algorithms and models come with design parameters that need to be set in advance. A widely adopted pratice is to search the parameters (usually through brute-force loops) that yields the best performance over a validation set. The package provides functions to facilitate this.

.. function:: gridtune(estfun, evalfun, params...; ...)

    Search the best setting of parameters over a Cartesian grid (*i.e.* all combinations of parameters).

    :param estfun: The model estimation function that takes design parameters as input and produces the model.
    :param evalfun: The function that evaluates the model, producing a score value.
    :param params: A series of parameters, given in the form of ``(param_name, param_values)``.

    :return: a 3-tuple, as ``(best_model, best_cfg, best_score)``. Here, ``best_cfg`` is a tuple comprised of the parameters in the best setting (the one that yields the best score).

    **Keyword arguments:**

    - ``ord``: It may take either of ``Forward`` or ``Reverse``:

        * ``ord=Forward``: higher score value indicates better model (default)
        * ``ord=Reverse``: lower score value indicates better model.

    - ``verbose``: boolean, whether to show progress information. (default = ``false``).


    **Note:** For some learning algorithms, there may be some constraint of the parameters (*e.g* one parameter must be smaller than another, etc). If a certain combination of parameters is not valid, the ``estfun`` may return nothing, in which case, the function would ignore those particular settings.

    **Example:**

    .. code-block:: julia

        using MLBase
        using MultivariateStats

        ## prepare data

        n_tr = 20  # number of training samples
        n_te = 10  # number of testing samples
        d = 5      # dimension of observations

        theta = randn(d)
        X_tr = randn(n_tr, d)
        y_tr = X_tr * theta + 0.1 * randn(n_tr)
        X_te = randn(n_te, d)
        y_te = X_te * theta + 0.1 * randn(n_te)

        ## tune the model

        function estfun(regcoef, bias)
            s = ridge(X_tr, y_tr, regcoef; bias=bias)
            return bias ? (s[1:end-1], s[end]) : (s, 0.0)
        end

        evalfun(m) = msd(X_te * m[1] + m[2], y_te) 

        r = gridtune(estfun, evalfun, 
                    ("regcoef", [1.0e-3, 1.0e-2, 1.0e-1, 1.0]), 
                    ("bias", (true, false)); 
                    ord=Reverse,    # smaller msd value indicates better model
                    verbose=true)   # show progress information

        best_model, best_cfg, best_score = r

        ## print results

        a, b = best_model
        println("Best model:") 
        println("  a = $(a')"), 
        println("  b = $b")
        println("Best config: regcoef = $(best_cfg[1]), bias = $(best_cfg[2])")
        println("Best score: $(best_score)")



