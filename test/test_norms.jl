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

# normalize

x = randn(5)
@test_approx_eq normalize(x, 2) x / norm(x, 2)
@test_approx_eq norm(normalize(x, 2), 2) 1.0
@test_approx_eq normalize(x, 1) x / norm(x, 1)
@test_approx_eq norm(normalize(x, 1), 1) 1.0 

xc = copy(x); normalize!(xc, 2)
@test_approx_eq xc normalize(x, 2)
xc = copy(x); normalize!(xc, 1)
@test_approx_eq xc normalize(x, 1)


x = randn(4, 5)
@test_approx_eq normalize(x, 2, 1) x ./ reshape(vnorm(x, 2, 1), 1, 5)
@test_approx_eq normalize(x, 2, 2) x ./ vnorm(x, 2, 2)

xc = copy(x); normalize!(xc, 2, 1)
@test_approx_eq xc normalize(x, 2, 1)
xc = copy(x); normalize!(xc, 2, 2)
@test_approx_eq xc normalize(x, 2, 2)


