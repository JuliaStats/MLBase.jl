using MLBase
using Test

m = 5
n = 6

x = rand(n)
X = rand(m, n)

# entropy

p = x / sum(x)
P1 = bsxfun(./, X, sum(X, 1))
P2 = bsxfun(./, X, sum(X, 2))

@test is_approx(entropy(p), -sum(p .* log(p)), 1.0e-12)
@test is_approx(entropy(P1, 1), -sum(P1 .* log(P1), 1), 1.0e-12)
@test is_approx(entropy(P2, 2), -sum(P2 .* log(P2), 2), 1.0e-12)


# logsumexp

@test is_approx(logsumexp(x), log(sum(exp(x))), 1.0e-12)
@test is_approx(logsumexp(X, 1), log(sum(exp(X), 1)), 1.0e-12)
@test is_approx(logsumexp(X, 2), log(sum(exp(X), 2)), 1.0e-12)

# softmax

r = softmax(x)
r0 = exp(x) / sum(exp(x))
@test is_approx(r, r0, 1.0e-12)

r = softmax(X, 1)
r0 = bsxfun(./, exp(X), sum(exp(X), 1))
@test is_approx(r, r0, 1.0e-12)

r = softmax(X, 2)
r0 = bsxfun(./, exp(X), sum(exp(X), 2))
@test is_approx(r, r0, 1.0e-12)
