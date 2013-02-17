using MLBase
using Test

m = 5
n = 6

x = rand(n)
X = rand(m, n)

@test_approx_eq logsumexp(x) log(sum(exp(x)))
@test_approx_eq logsumexp(X, 1) log(sum(exp(X), 1))
@test_approx_eq logsumexp(X, 2) log(sum(exp(X), 2))

@test_approx_eq softmax(x) exp(x) / sum(exp(x))
@test_approx_eq softmax(X, 1) bsxfun(./, exp(X), sum(exp(X), 1))
@test_approx_eq softmax(X, 2) bsxfun(./, exp(X), sum(exp(X), 2))

