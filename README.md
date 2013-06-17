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

wcounts2(m, n, x, y, w)   # returns a 2D counting matrix
```

#### Index arrangement

In machine learning, specially supervised learning, it is often needed to find the indices of samples corresponding to each class. The functions as described below would be helpful.

```julia
k = 3               # the number of classes
labels = [1, 1, 2, 2, 2, 3, 3, 1, 1]    

z, c = sort_indices(k, labels)    # explained below

println(z)     # ==> [1, 2, 8, 9, 3, 4, 5, 6, 7]
println(cnts)  # ==> [4, 3, 2]
```

Here, ``z`` is the vector of sorted indices, and ``cnts`` are the counts of each label. Then, ``z[1:4] == [1, 2, 8, 9]`` are the indices corresponding to label ``1``; ``z[5:7] == [3, 4, 5]`` are the indices for label ``2``; and ``z[8:9] == [6, 7]`` are the indices for label ``3``. One can make this grouping explicit using ``group_indices`` as below:

```julia
g = group_indices(k, labels)   # returns a vector comprised of three vectors

g[1]   # ==> [1, 2, 8, 9]
g[2]   # ==> [3, 4, 5]
g[3]   # ==> [6, 7]

 # the above is equivalent to
z, c = sort_indices(k, labels)
g = sorted_indices_to_groups(z, c)
```

#### Repeating numbers

The following function generates elements based on counts (kind of like the *inverse* of counting):

```julia
repeat_eachelem(x, c)      # generate a vector comprised of each element in x, 
                           # each repeated for specified number of times

repeat_eachelem(1:3, 4)        # ==> [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
repeat_eachelem(1:3, [1,2,3])  # ==> [1, 2, 2, 3, 3, 3]

repeat_eachelem([1.2, 2.3], [3, 2])  # ==> [1.2, 1.2, 1.2, 2.3, 2.3]
```

This function is sometimes useful in generating a sequence of labels.


## Computation on positive definite matrix

Positive definite matrices are widely used in machine learning and probabilistic modeling, especially in applications related to graph analysis and Gaussian models. It is not uncommon that positive definite matrices used in practice have special structures (e.g. diagonal), which can be exploited to accelerate computation. 

This package defines an abstract type ``AbstractPDMat`` to capture positive definite matrices of various structures, as well as three concrete sub-types: ``PDMat``, ``PDiagMat``, ``ScalMat``, which can be constructed as follows

```julia
PDMat(C)         # a wrapper of the positive definite matrix C, as a sub-type of AbstractPDMat
PDiagMat(v)      # corresponds to diagm(v)
Scalmat(d, v)    # corresponds to v * eye(d)
```

**Note:** Compact representation is used internally. For example, an instance of ``PDiagMat`` only contains a vector of diagonal elements instead of the full diagonal matrix, and ``ScalMat`` only contains a scalar value. While, for ``PDMat``, a Cholesky factorization is computed and contained in the instance for efficient computation.

Let ``a`` be an instance of one of the classes above, one can perform following operation on ``a``:

```julia
dim(a)       # the dimension of the matrix. If it is a d x d matrix, this returns d
full(a)      # the full matrix
inv(a)       # the inverse (which is still an instance of the same type)
logdet(a)    # log-determinant

a * x          # matrix-vector/matrix multiplication when x is a vector/matrix
a \ x          # equivalent to inv(a) * x, but implemented in a more efficient way
a * c, c * a   # multiply a by a scalar (the result is of the same type)

unwhiten(a, x)   # unwhiten transform, if x satisfies a standard Gaussian distribution,
                 # then unwhiten(a, x) has a distribution of covariance a
                 # Here, x can be either a vector or a matrix

whiten(a, x)     # inverse operation w.r.t. unwhiten

unwhiten!(a, x)  # inplace unwhiten
whiten!(a, x)    # inplace whiten

quad(a, x)       # compute x' * a * x in an efficient way, x can be a vector or a matrix
                 # if x is a matrix, it perform column-wise computation
                 # that is, it returns r, with r[i] == x[:,i]' * a * x[:,i]

invquad(a, x)    # compute x' * inv(a) * x in an efficient way, x can be a vector or a matrix
                 # if x is a matrix, it performs column-wise computation

quad!(r, a, x)      # inplace column-wise quadratic form computation for matrix x
invquad!(r, a, x)   # inplace column-wise quadratic form computation (w.r.t. inv(a)) for matrix x

X_A_Xt(a, x)        # computes x * a * x' for matrix x
Xt_A_X(a, x)        # computes x' * a * x for matrix x
X_invA_Xt(a, x)     # computes x * inv(a) * x' for matrix x
Xt_invA_X(a, x)     # computes x' * inv(a) * x for matrix x

a1 + a2          # add two positive definite matrices (promoted to a proper type)
a + m            # add a positive definite matrix and an ordinary matrix (returns an ordinary matrix)

add!(m, a)           # add the positive definite matrix a to an ordinary matrix m (inplace)
add_scal!(m, a, c)   # add a scaled version a * c to an ordinary matrix m (inplace)
add_scal(a1, a2, c)  # returns a1 + a2 * c (promoted to a proper type)
```

Specialized version of each of these functions are implemented for each specific postive matrix types using the most efficient routine (depending on the corresponding structures.)

**Note:** This framework provides uniform interfaces to use positive definite matrices of various structures for writing generic algorithms, while ensuring that the most efficient implementation is used in actual computation.


