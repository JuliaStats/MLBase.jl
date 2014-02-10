## MLBase.jl

[![Build Status](https://travis-ci.org/JuliaStats/MLBase.jl.png)](https://travis-ci.org/JuliaStats/MLBase.jl)

Basic functionalities for Machine Learning, including:

- Data manipulation
- Basic classification
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

### Basic Classification

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



