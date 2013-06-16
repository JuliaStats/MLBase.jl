# Test norms

using MLBase
using Base.Test

x = randn(7, 8)
y = randn(7, 8)

# vnorm

@test_approx_eq vnorm(x, 1, 1) vec(sum(abs(x), 1))
@test_approx_eq vnorm(x, 1, 2) vec(sum(abs(x), 2))

@test_approx_eq vnorm(x, 2, 1) vec(sqrt(sum(abs2(x), 1)))
@test_approx_eq vnorm(x, 2, 2) vec(sqrt(sum(abs2(x), 2)))

@test_approx_eq vnorm(x, Inf, 1) vec(max(abs(x), (), 1))
@test_approx_eq vnorm(x, Inf, 2) vec(max(abs(x), (), 2))

@test_approx_eq vnorm(x, 3., 1) vec(sum(abs(x) .^ 3, 1) .^ (1/3))
@test_approx_eq vnorm(x, 3., 2) vec(sum(abs(x) .^ 3, 2) .^ (1/3))

# vdiffnorm

@test_approx_eq vdiffnorm(x, y, 1, 1) vec(sum(abs(x - y), 1))
@test_approx_eq vdiffnorm(x, y, 1, 2) vec(sum(abs(x - y), 2))

@test_approx_eq vdiffnorm(x, y, 2, 1) vec(sqrt(sum(abs2(x - y), 1)))
@test_approx_eq vdiffnorm(x, y, 2, 2) vec(sqrt(sum(abs2(x - y), 2)))

@test_approx_eq vdiffnorm(x, y, Inf, 1) vec(max(abs(x - y), (), 1))
@test_approx_eq vdiffnorm(x, y, Inf, 2) vec(max(abs(x - y), (), 2))

@test_approx_eq vdiffnorm(x, y, 3., 1) vec(sum(abs(x - y) .^ 3, 1) .^ (1/3))
@test_approx_eq vdiffnorm(x, y, 3., 2) vec(sum(abs(x - y) .^ 3, 2) .^ (1/3))
