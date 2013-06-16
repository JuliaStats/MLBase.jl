
using MLBase
using Base.Test

# Auxiliary tools for testing

safe_quad(a::AbstractPDMat, x::Vector) = dot(x, full(a) * x)
safe_quad(a::AbstractPDMat, x::Matrix) = vec(sum(x .* (full(a) * x), 1))
safe_invquad(a::AbstractPDMat, x::Vector) = dot(x, full(a) \ x)
safe_invquad(a::AbstractPDMat, x::Matrix) = vec(sum(x .* (full(a) \ x), 1))

### PDMat ###

C = [4. -2. -1.; -2. 5. -1.; -1. -1. 6.]

# basics

a = PDMat(C)
@test a.mat === C
@test dim(a) == 3
@test full(a) == C
@test !is(full(a), C)
@test_approx_eq logdet(a) log(det(C))

r = inv(a)
@test isa(r, PDMat)
@test dim(r) == 3
@test_approx_eq r.mat inv(a.mat)

# multiplication and solve

x = rand(3, 5)
xt = x'
x1 = rand(3)

@test_approx_eq a * x full(a) * x
@test_approx_eq a \ x full(a) \ x

# whiten & unwhiten

u = unwhiten(a, eye(3))
@test_approx_eq u * u' full(a)
@test_approx_eq whiten(a, eye(3)) inv(u)

y = unwhiten(a, x)
xc = copy(x)
unwhiten!(a, xc)
@test y == xc
@test_approx_eq y transpose(chol(a.mat)) * x

xr = whiten(a, y)
yc = copy(y)
whiten!(a, yc)
@test yc == xr
@test_approx_eq xr x
@test_approx_eq unwhiten(a, whiten(a, x)) x

# quad & invquad

@test_approx_eq quad(a, x1) safe_quad(a, x1)
@test_approx_eq quad(a, x) safe_quad(a, x)
@test_approx_eq invquad(a, x1) safe_invquad(a, x1)
@test_approx_eq invquad(a, x) safe_invquad(a, x)

r = xt * full(a) * x
@test_approx_eq Xt_A_X(a, x) r
@test_approx_eq X_A_Xt(a, xt) r
r = xt * (full(a) \ x)
@test_approx_eq Xt_invA_X(a, x) r
@test_approx_eq X_invA_Xt(a, xt) r


### PDiagMat ###

va = [1.5, 2.5, 2.0]

# basics

a = PDiagMat(va)
@test a.diag === va
@test dim(a) == 3
@test_approx_eq full(a) diagm(va)
@test_approx_eq logdet(a) log(det(full(a)))

r = inv(a)
@test isa(r, PDiagMat)
@test dim(r) == 3
@test r.diag == 1.0 ./ va

# mult and solve

x = rand(3, 5)
xt = x'
x1 = rand(3)

@test_approx_eq a * x full(a) * x
@test_approx_eq a \ x full(a) \ x

# whiten and unwhiten

u = unwhiten(a, eye(3))
@test_approx_eq u * u' full(a)
@test_approx_eq whiten(a, eye(3)) inv(u)

y = unwhiten(a, x)
xc = copy(x)
unwhiten!(a, xc)
@test y == xc
@test_approx_eq y bsxfun(.*, sqrt(a.diag), x)

xr = whiten(a, y)
yc = copy(y)
whiten!(a, yc)
@test yc == xr
@test_approx_eq xr x
@test_approx_eq unwhiten(a, whiten(a, x)) x

# quad and invquad

@test_approx_eq quad(a, x1) safe_quad(a, x1)
@test_approx_eq quad(a, x) safe_quad(a, x)
@test_approx_eq invquad(a, x1) safe_invquad(a, x1)
@test_approx_eq invquad(a, x) safe_invquad(a, x)

r = xt * full(a) * x
@test_approx_eq Xt_A_X(a, x) r
@test_approx_eq X_A_Xt(a, xt) r
r = xt * (full(a) \ x)
@test_approx_eq Xt_invA_X(a, x) r
@test_approx_eq X_invA_Xt(a, xt) r


### ScalMat ###

# basics

dv = 2.0

a = ScalMat(3, dv)
@test a.value == dv
@test dim(a) == 3
@test_approx_eq full(a) dv * eye(3,3)
@test_approx_eq logdet(a) log(det(full(a)))

r = inv(a)
@test isa(r, ScalMat)
@test dim(r) == 3
@test r.value == inv(dv)

# mult and solve

x = rand(3, 5)
xt = x'
x1 = rand(3)

@test_approx_eq a * x full(a) * x
@test_approx_eq a \ x full(a) \ x

# whiten and unwhiten

u = unwhiten(a, eye(3))
@test_approx_eq u * u' full(a)
@test_approx_eq whiten(a, eye(3)) inv(u)

y = unwhiten(a, x)
xc = copy(x)
unwhiten!(a, xc)
@test y == xc
@test_approx_eq y sqrt(a.value) * x

xr = whiten(a, y)
yc = copy(y)
whiten!(a, yc)
@test yc == xr
@test_approx_eq xr x
@test_approx_eq unwhiten(a, whiten(a, x)) x

# quad and invquad

@test_approx_eq quad(a, x1) safe_quad(a, x1)
@test_approx_eq quad(a, x) safe_quad(a, x)
@test_approx_eq invquad(a, x1) safe_invquad(a, x1)
@test_approx_eq invquad(a, x) safe_invquad(a, x)

r = xt * full(a) * x
@test_approx_eq Xt_A_X(a, x) r
@test_approx_eq X_A_Xt(a, xt) r
r = xt * (full(a) \ x)
@test_approx_eq Xt_invA_X(a, x) r
@test_approx_eq X_invA_Xt(a, xt) r




