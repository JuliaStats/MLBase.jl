
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

@test_approx_eq full(a * 2.0) full(a) * 2.0
@test_approx_eq full(a / 2.0) full(a) / 2.0
@test full(a * 2.0) == full(2.0 * a)

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

@test_approx_eq full(a * 2.0) full(a) * 2.0
@test_approx_eq full(a / 2.0) full(a) / 2.0
@test full(a * 2.0) == full(2.0 * a)

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

@test_approx_eq full(a * 2.0) full(a) * 2.0
@test_approx_eq full(a / 2.0) full(a) / 2.0
@test full(a * 2.0) == full(2.0 * a)

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


### Addition ###

va = [1.5, 2.5, 2.0]
C = [4. -2. -1.; -2. 5. -1.; -1. -1. 6.]

pm = PDMat(copy(C))
pd = PDiagMat(copy(va))
ps = ScalMat(3, 2.0)

A = rand(3, 3)

# bewteen pdmat and mat

@test_approx_eq pm + A full(pm) + A
@test_approx_eq pd + A full(pd) + A
@test_approx_eq ps + A full(ps) + A

@test (pm + A) == (A + pm)
@test (pd + A) == (A + pd)
@test (ps + A) == (A + ps)

Ac = copy(A); add!(Ac, pm) 
@test_approx_eq Ac A + pm

Ac = copy(A); add!(Ac, pd) 
@test_approx_eq Ac A + pd

Ac = copy(A); add!(Ac, ps) 
@test_approx_eq Ac A + ps

@test_approx_eq add_scal(A, pm, 2.) A + full(pm) * 2.
@test_approx_eq add_scal(A, pd, 2.) A + full(pd) * 2.
@test_approx_eq add_scal(A, ps, 2.) A + full(ps) * 2.

# between pd-matrices

@test isa(pm + pm, PDMat)
@test_approx_eq full(pm + pm) full(pm) + full(pm)

@test isa(pm + pd, PDMat)
@test_approx_eq full(pm + pd) full(pm) + full(pd)

@test isa(pm + ps, PDMat)
@test_approx_eq full(pm + ps) full(pm) + full(ps)

@test isa(pd + pm, PDMat)
@test_approx_eq full(pd + pm) full(pd) + full(pm)

@test isa(pd + pd, PDiagMat)
@test_approx_eq full(pd + pd) full(pd) + full(pd)

@test isa(pd + ps, PDiagMat)
@test_approx_eq full(pd + ps) full(pd) + full(ps)

@test isa(ps + pm, PDMat)
@test_approx_eq full(ps + pm) full(ps) + full(pm)

@test isa(ps + pd, PDiagMat)
@test_approx_eq full(ps + pd) full(ps) + full(pd)

@test isa(ps + ps, ScalMat)
@test_approx_eq full(ps + ps) full(ps) + full(ps)

@test isequal(pm.mat, C)
@test isequal(pd.diag, va)

# add scaled

r = add_scal(pm, pm, 2.0)
@test isa(r, PDMat)
@test_approx_eq full(r) full(pm) + full(pm) * 2.0

r = add_scal(pm, pd, 2.0)
@test isa(r, PDMat)
@test_approx_eq full(r) full(pm) + full(pd) * 2.0

r = add_scal(pm, ps, 2.0)
@test isa(r, PDMat)
@test_approx_eq full(r) full(pm) + full(ps) * 2.0

r = add_scal(pd, pm, 2.0)
@test isa(r, PDMat)
@test_approx_eq full(r) full(pd) + full(pm) * 2.0

r = add_scal(pd, pd, 2.0)
@test isa(r, PDiagMat)
@test_approx_eq full(r) full(pd) + full(pd) * 2.0

r = add_scal(pd, ps, 2.0)
@test isa(r, PDiagMat)
@test_approx_eq full(r) full(pd) + full(ps) * 2.0

r = add_scal(ps, pm, 2.0)
@test isa(r, PDMat)
@test_approx_eq full(r) full(ps) + full(pm) * 2.0

r = add_scal(ps, pd, 2.0)
@test isa(r, PDiagMat)
@test_approx_eq full(r) full(ps) + full(pd) * 2.0

r = add_scal(ps, ps, 2.0)
@test isa(r, ScalMat)
@test_approx_eq full(r) full(ps) + full(ps) * 2.0


