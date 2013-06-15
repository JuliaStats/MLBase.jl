# Test vector statistics

using MLBase
using Base.Test

x = randn(7, 8)

# vsum
@test_approx_eq vsum(x, 1) vec(sum(x, 1))
@test_approx_eq vsum(x, 2) vec(sum(x, 2))

r = zeros(size(x, 2))
vsum!(r, x, 1) 
@test r == vsum(x, 1)

r = zeros(size(x, 1))
vsum!(r, x, 2)
@test r == vsum(x, 2)

@test_approx_eq vsum(zeros(0, 4), 1) zeros(4)
@test_approx_eq vsum(zeros(4, 0), 2) zeros(4)
@test_approx_eq vsum(reshape([1., 2., 3.], 1, 3), 1) [1., 2., 3.]
@test_approx_eq vsum(reshape([1., 2., 3.], 3, 1), 2) [1., 2., 3.] 

# vmax & vmin
@test_approx_eq vmax(x, 1) vec(max(x, (), 1))
@test_approx_eq vmax(x, 2) vec(max(x, (), 2))

r = zeros(size(x, 2))
vmax!(r, x, 1) 
@test r == vmax(x, 1)

r = zeros(size(x, 1))
vmax!(r, x, 2)
@test r == vmax(x, 2)

@test_approx_eq vmin(x, 1) vec(min(x, (), 1))
@test_approx_eq vmin(x, 2) vec(min(x, (), 2))

r = zeros(size(x, 2))
vmin!(r, x, 1) 
@test r == vmin(x, 1)

r = zeros(size(x, 1))
vmin!(r, x, 2)
@test r == vmin(x, 2)

# vasum, vamax, vamin, vsqsum

@test_approx_eq vasum(x, 1) vec(sum(abs(x), 1))
@test_approx_eq vasum(x, 2) vec(sum(abs(x), 2))

@test_approx_eq vamax(x, 1) vec(max(abs(x), (), 1))
@test_approx_eq vamax(x, 2) vec(max(abs(x), (), 2))

@test_approx_eq vamin(x, 1) vec(min(abs(x), (), 1))
@test_approx_eq vamin(x, 2) vec(min(abs(x), (), 2))

@test_approx_eq vsqsum(x, 1) vec(sum(abs2(x), 1))
@test_approx_eq vsqsum(x, 2) vec(sum(abs2(x), 2))

