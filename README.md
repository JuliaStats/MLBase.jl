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

    Repeat each element in vector ``a`` for ``n`` times. 

    ```julia
    julia> using MLBase

    julia> repeach(1:3, 2)
    6-element Array{Int64,1}:
     1
     1
     2
     2
     3
     3
    ```

