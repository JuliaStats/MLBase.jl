# Tests of intstats.jl

using MLBase
using Base.Test

# classify

ss = rand(8, 50)
for i = 1:size(ss,2)
    ss_i = ss[:,i]
    kmax = indmax(ss_i)
    kmin = indmin(ss_i)
    vmax = ss_i[kmax]
    vmin = ss_i[kmin]

    @test classify(ss_i) == kmax
    @test classify(ss_i, Forward) == kmax
    @test classify(ss_i, Reverse) == kmin

    @test classify(ss_i, 0.8) == (vmax >= 0.8 ? kmax : 0)
    @test classify(ss_i, 0.8, Forward) == (vmax >= 0.8 ? kmax : 0)
    @test classify(ss_i, 0.2, Reverse) == (vmin <= 0.2 ? kmin : 0)

    @test classify_withscore(ss_i) == (kmax, ss_i[kmax])
    @test classify_withscore(ss_i, Forward) == (kmax, ss_i[kmax])
    @test classify_withscore(ss_i, Reverse) == (kmin, ss_i[kmin])
end

rmax = Int[indmax(ss[:,i]) for i = 1:size(ss,2)]
rmin = Int[indmin(ss[:,i]) for i = 1:size(ss,2)]
vmax = ss[sub2ind(size(ss), rmax, 1:size(ss,2))]
vmin = ss[sub2ind(size(ss), rmin, 1:size(ss,2))]

trmax = copy(rmax); trmax[vmax .< 0.8] = 0
trmin = copy(rmin); trmin[vmin .> 0.2] = 0

@test classify(ss) == rmax
@test classify(ss, Forward) == rmax
@test classify(ss, Reverse) == rmin

@test classify_withscores(ss) == (rmax, vmax)
@test classify_withscores(ss, Forward) == (rmax, vmax)
@test classify_withscores(ss, Reverse) == (rmin, vmin)

@test classify(ss, 0.8) == trmax
@test classify(ss, 0.8, Forward) == trmax
@test classify(ss, 0.2, Reverse) == trmin

# labelmap & labelencode

xs = ["a", "a", "b", "b", "a", "b", "c", "a"]
labels = [1, 1, 2, 2, 1, 2, 3, 1]
lmap = labelmap(xs)

@test keys(lmap) == ["a", "b", "c"]
@test labelencode(lmap, xs) == labels
@test labeldecode(lmap, labels) == xs

# groupindices

gs = Any[[1,2,5,8],[3,4,6],[7]]

@test groupindices(3, labels) == gs
@test groupindices(lmap, xs) == gs

