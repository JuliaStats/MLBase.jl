## MLBase.jl

A collection of tools to support the implementation of machine learning algorithms.

This package provides functions in following categories, which are commonly used in machine learning:
* Efficient matrix-vector arithmetic (e.g. adding/subtracting/multiplying a vector to each row/column of a matrix)
* Efficient column-wise or row-wise reduction
* Computation of column-wise or row-wise norms, and normalization
* Positive definite matrix related computation
* Integer related statistics

More functions may be added in the future as needed. 

## Inplace vector arithmetics

```julia
add!(x, y)     # add y to x
sub!(x, y)     # subtract y from x
mul!(x, y)     # multiply y to x

rcp!(x)        # compute the reciprocal (i.e. 1/x) of each element (inplace)
sqrt!(x)       # compute the square root of each element inplace
exp!(x)        # compute exponentiation of each element inplace
log!(x)        # compute logarithm of each element inplace
```

**Note:** there is an important difference between ``add!(x, y)`` and ``x += y``. In Julia, ``x += y`` will create a new array that contains the sum and rebounds ``x`` to this new array; while ``add!(x, y)`` directly adds ``y`` to ``x`` without new matrix creation. The same applies to ``sub!`` and ``mul!``.











