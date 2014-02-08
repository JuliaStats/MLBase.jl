# Tests of intstats.jl

using MLBase
using Base.Test

# classify

ss = rand(8, 50)
for i = 1:size(ss,2)
	@test classify(ss[:,i]) == indmax(ss[:,i])
	@test classify(ss[:,i]; to_max=false) == indmin(ss[:,i])
end

rmax = Int[indmax(ss[:,i]) for i = 1:size(ss,2)]
rmin = Int[indmin(ss[:,i]) for i = 1:size(ss,2)]
@test classify(ss) == rmax
@test classify(ss; to_max=false) == rmin

# thresholded classify

maxs = vec(maximum(ss, 1))
mins = vec(minimum(ss, 1))

trmax = rmax; trmax[maxs .< 0.8] = 0
trmin = rmin; trmin[mins .> 0.2] = 0

for i = 1:size(ss,2)
	@test thresholded_classify(ss[:,i], 0.8) == trmax[i]
	@test thresholded_classify(ss[:,i], 0.2; to_max=false) == trmin[i]
end

@test thresholded_classify(ss, 0.8) == trmax
@test thresholded_classify(ss, 0.2; to_max=false) == trmin

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

