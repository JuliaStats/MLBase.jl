# Tests of sampling tools

using MLBase
using Test

# sample_by_weights

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

# sample_without_replacement

k = 3
n = 1000000
cnts = zeros(Int, 10)

for i = 1 : n
	r = sample_without_replacement(1:10, k)
	@assert r[1] != r[2] && r[1] != r[3] && r[2] != r[3]
	for j = 1 : k
		cnts[r[j]] += 1
	end
end
p = cnts / (n * k)
p0 = fill(0.1, 10)
@test all(abs(p - p0) .< 0.01)
