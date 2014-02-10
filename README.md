## MLBase.jl

[![Build Status](https://travis-ci.org/JuliaStats/MLBase.jl.png)](https://travis-ci.org/JuliaStats/MLBase.jl)

Basic functionalities for Machine Learning, including:

- Data manipulation
- Simple classification
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





