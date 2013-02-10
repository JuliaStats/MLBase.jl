# Tests of sampling tools

using MLBase
using Test

w = [1., 2., 3., 4.]
n = 1000000

cnts = zeros(Int, 4)
for i = 1 : n
	xi = sample_by_weights(w, 10.)
	cnts[xi] += 1
end
p = cnts / n
p0 = w / sum(w)
@test all(abs(p - p0) .< 0.01)
