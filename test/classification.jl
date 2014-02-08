# Tests of intstats.jl

using MLBase
using Base.Test

# labelmap & labelencode

xs = ["a", "a", "b", "b", "a", "b", "c", "a"]
labels = [1, 1, 2, 2, 1, 2, 3, 1]
lmap = labelmap(xs)

@test keys(lmap) == ["a", "b", "c"]
@test labelencode(lmap, xs) == labels

# groupindices

gs = {[1,2,5,8],[3,4,6],[7]}

@test groupindices(3, labels) == gs
@test groupindices(lmap, xs) == gs

