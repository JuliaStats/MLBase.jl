# Tests of intstats.jl

using MLBase
using Base.Test

# repeach

@test repeach(2:4, 2) == [2, 2, 3, 3, 4, 4]
@test repeach(2:4, 1:3) == [2, 3, 3, 4, 4, 4]

# repeachcol

a = rand(4, 3)

@test repeachcol(a, 2) == a[:, [1,1,2,2,3,3]]
@test repeachcol(a, 1:3) == a[:, [1,2,2,3,3,3]]

# repeachrow

a = rand(3, 4)
@test repeachrow(a, 2) == a[[1,1,2,2,3,3], :]
@test repeachrow(a, 1:3) == a[[1,2,2,3,3,3], :]

