## MLBase.jl

A collection of tools to support the implementation of machine learning algorithms. More functions may be added in the future as needed. 

**Note:** A large portion of this package that is related to vectorized computation has been migrated to (NumericExtensions.jl)[https://github.com/lindahua/NumericExtensions.jl]. This package only maintains the part that is specific to machine learning.


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

