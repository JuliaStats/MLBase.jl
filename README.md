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


## Broadcasting vector arithmetics

```julia
add_cols!(x, v)    # add vector v to each column of x inplace
add_rows!(x, v)    # add vector v to each row of x inplace
add_cols(x, v)     # return a new matrix y, where y[:,i] == x[:,i] + v
add_rows(x, v)     # return a new matrix y, where y[i,:] == x[i,:] + v

sub_cols!(x, v)    # subtract vector v from each column of x inplace
sub_rows!(x, v)    # subtract vector v from each row of x inplace
sub_cols(x, v)     # return a new matrix y, where y[:,i] == x[:,i] - v
sub_rows(x, v)     # return a new matrix y, where y[i,:] == x[i,:] - v

mul_cols!(x, v)    # multiply vector v to each column of x inplace
mul_rows!(x, v)    # multiply vector v to each row of x inplace
mul_cols(x, v)     # return a new matrix y, where y[:,i] == x[:,i] .* v
mul_rows(x, v)     # return a new matrix y, where y[i,:] == x[i,:] .* v
```

## Vector-wise reduction

```julia
vsum!(r, x, dim)   # sum along a specific dimension and write results to r
vsum(x, dim)       # return colwise sums (dim = 1) or rowwise sums (dim = 2)
```

**Note:** suppose ``x`` is a matrix of size ``(m, n)``, then ``vsum(x, dim)`` is a vector of length ``n`` when ``dim == 1``, or a vector length ``m`` when ``dim == 2``. This is different from ``sum(x, dim)``, where both both ``sum(x, 1)`` and ``sum(x, 2)`` return a matrix (respectively of size ``(1, n)`` and ``(m, 1)``).

In addition to sum, this package provides a series of vector reduction functions, as follows

```julia
vmean!(r, x, dim), vmean(x, dim)         # mean
vmax!(r, x, dim), vmax(x, dim)           # maximum
vmin!(r, x, dim), vmin(x, dim)           # minimum
vasum!(r, x, dim), vasum(x, dim)         # sum of absolute value
vamax!(r, x, dim), vamax(x, dim)         # maximum of absolute value
vamin!(r, x, dim), vamin(x, dim)         # minimum of absolute value
vsqsum!(r, x, dim), vsqsum(x, dim)       # sum of square

vpowsum!(r, x, p, dim), vpowsum(x, p, dim)        # sum of power (i.e. x^p)
vdot!(r, x, y, dim), vdot(x, y, dim)              # colwise/rowwise dot product
vsqdiffsum!(r, x, y, dim), vsqdiffsum(x, y, dim)  # sum of squared differences
```

**Note:** These functions are implemented efficiently. In particular, (1) all computation are carefully de-vectorized and thus no intermediate arrays are created, and (2) the computation is conducted in a cache-friendly order. This leads to considerably faster code than vectorized Julia expressions. For example, ``vasum(x, 2)`` is about ``9x`` faster than ``sum(abs(x), 2)``.

The script ``test/bench_vreduc`` performs benchmarks that compare the performance of these functions with typical Julia vectorized expressions. Below are the results we obtained on a MacBook Pro (with *Julia v0.2.0*, each run over ``2000 x 2000`` matrices for ``9`` times):

| MLBase function | Julia expression    | gain | MLBase function | Julia expression    | gain |
|-----------------|---------------------|------|-----------------|---------------------|------|
| vsum(x, 1)      | sum(x, 1)           | 1.05 | vsum(x, 2)      | sum(x, 2)           | 5.84 |
| vmean(x, 1)     | mean(x, 1)          | 1.04 | vmean(x, 2)     | mean(x, 2)          | 5.96 |
| vmax(x, 1)      | max(x, (), 1)       | 1.80 | vmax(x, 2)      | max(x, (), 2)       | 3.07 |
| vmin(x, 1)      | min(x, (), 1)       | 1.73 | vmin(x, 2)      | min(x, (), 2)       | 3.37 |
| vasum(x, 1)     | sum(abs(x), 1)      | 5.25 | vasum(x, 2)     | sum(abs(x), 2)      | 9.15 |
| vamax(x, 1)     | max(abs(x), (), 1)  | 3.16 | vamax(x, 2)     | max(abs(x), (), 2)  | 4.46 |
| vamin(x, 1)     | min(abs(x), (), 1)  | 3.18 | vamin(x, 2)     | min(abs(x), (), 2)  | 4.40 |
| vsqsum(x, 1)    | sum(abs2(x), 1)     | 6.46 | vsqsum(x, 2)    | sum(abs2(x), 2)     | 9.28 |
| vdot(x, y, 1)   | sum(x .* y, 1)      | 5.15 | vdot(x, y, 2)   | sum(x .* y, 2)      | 8.91 |
| vsqdiffsum(x, y, 1) | sum(abs2(x - y), 1) | 3.39 | vsqdiffsum(x, y, 2) | sum(abs2(x - y), 2) | 5.11 |

Here, when the *gain* is greater than 1, it means that the MLBase function is faster than the Julia vectorized expression.

The reduction is implemented based on a generic framework (see ``src/vecreduc.jl``) and thus can be easily extended. Also, this package does not provide functions to perform reduction over a single vector, because the reduction functions in Julia are efficient enough for single vectors.

## Vector-norms and normalization

Built on top the efficient reduction framework, this package also provides functions to compute vector norms and perform normalization.

```julia
vnorm(x, p, dim)          # compute L-p norms for columns(or rows) in x
vdiffnorm(x, y, p, dim)   # compute L-p norms for differences between columns/rows in x and y
```

**Note:** When ``p`` is ``1``, ``2``, or ``Inf``, specialized routines are used for efficient computation. Also, ``vdiffnorm(x, y, p, dim)`` is functionally equivalent to ``vnorm(x - y, p, dim)``, but it uses de-vectorized implementation and does not explicitly construct the matrix ``x - y``, and thus is more efficient.

```julia
normalize!(x, p)        # normalize a vector inplace (by L-p norm)
normalize(x, p)         # returns a normalized vector (by L-p norm)

normalize!(x, p, dim)   # normalize each column/row in place
normalize(x, p, dim)    # returns a new matrix comprised of normalized columns/rows
```


## Integer-related Tools

#### Integer counting

Machine learning algorithms often requires handling integer indices or labels. This package provides some functions to facilitate such tasks.

```julia
icounts(k, x)    # return a counting vector c of length k, 
                 # such that c[i] equals nnz(x == i), i.e. the number of times i appears in x

icounts(3, [1, 2, 2, 3, 3, 3, 3])  # ==> [1, 3, 4]

icounts2(m, n, x, y)   # returns a counting matrix of size (m, n),
                       # such that c[i,j] equals nnz((x .== i) && (y .== j))

icounts2(2, 2, [1, 1, 2, 2, 1], [1, 1, 1, 2, 2]) #=> [2 1; 1 1]
```

One can also add weights to each sample, as

```julia
wcounts(k, x, w)    # return a weighted counting vector for x

wcounts(2, [1, 1, 2], [3.0, 2.0, 1.0])  # ===> [5.0, 1.0]

wcounts(m, n, x, y, w)   # returns a 2D counting matrix
```





















