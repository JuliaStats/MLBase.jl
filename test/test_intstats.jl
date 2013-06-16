# Tests of intstats.jl

using MLBase
using Base.Test

# integer counting

x = rand(1:5, 200)
c = icounts(5, x)
c0 = Int[nnz(x .== i) for i in 1 : 5]
@test c == c0

x = rand(1:4, 500)
y = rand(1:5, 500)
c = icounts2(4, 5, x, y)
c0 = Int[nnz((x .== i) & (y .== j)) for i in 1 : 4, j in 1 : 5]
@test c == c0

x = rand(1:5, 200)
w = rand(200)
c = wcounts(5, x, w)
c0 = Float64[sum(w[x .== i]) for i in 1 : 5]
@test_approx_eq c c0

x = rand(1:4, 500)
y = rand(1:5, 500)
w = rand(500)
c = wcounts2(4, 5, x, y, w)
c0 = Float64[sum(w[(x .== i) & (y .== j)]) for i in 1 : 4, j in 1 : 5]
@test_approx_eq c c0

# indices arrangement

x = [1, 1, 1, 2, 2, 3, 3, 3, 3, 1, 1, 2, 1]
sx, cx = sort_indices(3, x)
@test sx == [1, 2, 3, 10, 11, 13, 4, 5, 12, 6, 7, 8, 9]
@test cx == [6, 3, 4]

g = group_indices(3, x)
@test size(g) == (3,)
@test g[1] == [1, 2, 3, 10, 11, 13]
@test g[2] == [4, 5, 12]
@test g[3] == [6, 7, 8, 9]

x = [3, 3, 3, 1, 1]
sx, cx = sort_indices(4, x)
@test sx == [4, 5, 1, 2, 3]
@test cx == [2, 0, 3, 0]

g = group_indices(4, x)
@test size(g) == (4,)
@test g[1] == [4, 5]
@test isempty(g[2])
@test g[3] == [1, 2, 3]
@test isempty(g[4])

# element repeating

@test repeat_eachelem(1:3, 4) == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
@test repeat_eachelem(1:3, [1,2,3]) == [1, 2, 2, 3, 3, 3]
@test repeat_eachelem(1:4, [0,3,0,4]) == [2, 2, 2, 4, 4, 4, 4]
