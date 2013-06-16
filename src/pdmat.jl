# Positive definite matrices of operations

abstract AbstractPDMat

#################################################
#
#	PDMat: full pos. def. matrix
#
#################################################

immutable PDMat <: AbstractPDMat
    dim::Int
    mat::Matrix{Float64}    
    chol::Cholesky{Float64}    
    
    function PDMat(mat::Matrix{Float64})
        d = size(mat, 1)
        if !(d >= 1 && size(mat, 2) == d)
            throw(ArgumentError("mat must be a square matrix."))
        end
        new(d, mat, cholfact(mat))
    end
end

# basics

dim(a::PDMat) = a.dim
full(a::PDMat) = copy(a.mat)
inv(a::PDMat) = PDMat(inv(a.chol))
logdet(a::PDMat) = logdet(a.chol)

* (a::PDMat, c::Float64) = PDMat(a.mat * c)
* (a::PDMat, x::VecOrMat) = a.mat * x
\ (a::PDMat, x::VecOrMat) = a.chol \ x

# whiten and unwhiten

whiten!(a::PDMat, x::VecOrMat{Float64}) = (trtrs!('U', 'T', 'N', a.chol.UL, x); x)
whiten(a::PDMat, x::VecOrMat{Float64}) = whiten!(a, copy(x))

unwhiten(a::PDMat, x::Vector{Float64}) = trmv('U', 'T', 'N', a.chol.UL, x)
unwhiten!(a::PDMat, x::Vector{Float64}) = (trmv!('U', 'T', 'N', a.chol.UL, x); x)

unwhiten(a::PDMat, x::Matrix{Float64}) = trmm('L', 'U', 'T', 'N', 1.0, a.chol.UL, x)
unwhiten!(a::PDMat, x::Matrix{Float64}) = (trmm!('L', 'U', 'T', 'N', 1.0, a.chol.UL, x); x)

# quadratic forms

quad(a::PDMat, x::Vector{Float64}) = dot(x, a.mat * x)
invquad(a::PDMat, x::Vector{Float64}) = abs2(norm(whiten(a, x)))
    
quad!(r::Array{Float64}, a::PDMat, x::Matrix{Float64}) = vdot!(r, x, a.mat * x, 1)
invquad!(r::Array{Float64}, a::PDMat, x::Matrix{Float64}) = vsqsum!(r, whiten(a, x), 1)

function X_A_Xt(a::PDMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 2)
    z = trmm('R', 'U', 'T', 'N', 1.0, a.chol.UL, x)
    gemm('N', 'T', 1.0, z, z)
end

function Xt_A_X(a::PDMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 1)
    z = trmm('L', 'U', 'N', 'N', 1.0, a.chol.UL, x)
    gemm('T', 'N', 1.0, z, z)
end

function X_invA_Xt(a::PDMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 2)
    z = transpose(x)
    trtrs!('U', 'T', 'N', a.chol.UL, z)
    gemm('T', 'N', 1.0, z, z)
end

function Xt_invA_X(a::PDMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 1)
    z = copy(x)
    trtrs!('U', 'T', 'N', a.chol.UL, z)
    gemm('T', 'N', 1.0, z, z)
end


#################################################
#
#	PDiagMat: positive diagonal matrix
#
#################################################

immutable PDiagMat <: AbstractPDMat
    dim::Int
    diag::Vector{Float64}
    inv_diag::Vector{Float64}
    
    PDiagMat(v::Vector{Float64}) = new(length(v), v, 1.0./v)    
    
    function PDiagMat(v::Vector{Float64}, inv_v::Vector{Float64})
        @check_argdims length(v) == length(inv_v)
        new(length(v), v, inv_v)
    end
end

# basics

dim(a::PDiagMat) = a.dim
full(a::PDiagMat) = diagm(a.diag)
inv(a::PDiagMat) = PDiagMat(a.inv_diag, a.diag)
logdet(a::PDiagMat) = sum(log(a.diag))

* (a::PDiagMat, c::Float64) = PDiagMat(a.diag * c)
* (a::PDiagMat, x::VecOrMat) = mul_cols(x, a.diag)
\ (a::PDiagMat, x::VecOrMat) = mul_cols(x, a.inv_diag)

# whiten and unwhiten 

function _mul_sqrt(x::Vector, c::Vector) 
	@check_argdims length(x) == length(c)
	[x[i] * sqrt(c[i]) for i in 1 : length(x)]
end

function _mul_sqrt!(x::Vector, c::Vector)
	@check_argdims length(x) == length(c)
	for i in 1 : length(x)
		x[i] .*= sqrt(c[i])
	end
	x
end

whiten(a::PDiagMat, x::Vector{Float64}) = _mul_sqrt(x, a.inv_diag)
whiten(a::PDiagMat, x::Matrix{Float64}) = mul_cols(x, sqrt(a.inv_diag))

whiten!(a::PDiagMat, x::Vector{Float64}) = _mul_sqrt!(x, a.inv_diag)
whiten!(a::PDiagMat, x::Matrix{Float64}) = mul_cols!(x, sqrt(a.inv_diag))

unwhiten(a::PDiagMat, x::Vector{Float64}) = _mul_sqrt(x, a.diag)
unwhiten(a::PDiagMat, x::Matrix{Float64}) = mul_cols(x, sqrt(a.diag))

unwhiten!(a::PDiagMat, x::Vector{Float64}) = _mul_sqrt!(x, a.diag)
unwhiten!(a::PDiagMat, x::Matrix{Float64}) = mul_cols!(x, sqrt(a.diag))

# quadratic forms

quad(a::PDiagMat, x::Vector{Float64}) = weighted_sqsum(x, a.diag)
invquad(a::PDiagMat, x::Vector{Float64}) = weighted_sqsum(x, a.inv_diag)

quad!(r::Array{Float64}, a::PDiagMat, x::Matrix{Float64}) = gemv!('T', 1., abs2(x), a.diag, 0., r)
invquad!(r::Array{Float64}, a::PDiagMat, x::Matrix{Float64}) = gemv!('T', 1., abs2(x), a.inv_diag, 0., r)

function X_A_Xt(a::PDiagMat, x::Matrix{Float64}) 
    z = mul_rows(x, sqrt(a.diag))
    gemm('N', 'T', 1.0, z, z)
end

function Xt_A_X(a::PDiagMat, x::Matrix{Float64})
    z = mul_cols(x, sqrt(a.diag))
    gemm('T', 'N', 1.0, z, z)
end

function X_invA_Xt(a::PDiagMat, x::Matrix{Float64})
    z = mul_rows(x, sqrt(a.inv_diag))
    gemm('N', 'T', 1.0, z, z)
end

function Xt_invA_X(a::PDiagMat, x::Matrix{Float64})
    z = mul_cols(x, sqrt(a.inv_diag))
    gemm('T', 'N', 1.0, z, z)
end


#################################################
#
#	ScalMat: s * eye(d) with s > 0
#
#################################################

immutable ScalMat <: AbstractPDMat
    dim::Int
    value::Float64
    inv_value::Float64
    
    ScalMat(d::Int, v::Float64) = new(d, v, 1.0 / v)
    ScalMat(d::Int, v::Float64, inv_v::Float64) = new(d, v, inv_v)
end

# basics

dim(a::ScalMat) = a.dim
full(a::ScalMat) = diagm(fill(a.value, a.dim))
inv(a::ScalMat) = ScalMat(a.dim, a.inv_value, a.value)
logdet(a::ScalMat) = a.dim * log(a.value)

* (a::ScalMat, c::Float64) = ScalMat(a.dim, a.value * c)
/ (a::ScalMat, c::Float64) = ScalMat(a.dim, a.value / c)
* (a::ScalMat, x::VecOrMat) = a.value * x
\ (a::ScalMat, x::VecOrMat) = a.inv_value * x

# whiten and unwhiten 

function whiten(a::ScalMat, x::VecOrMat{Float64})
    @check_argdims dim(a) == size(x, 1)
    x * sqrt(a.inv_value)
end

function whiten!(a::ScalMat, x::VecOrMat{Float64})
    @check_argdims dim(a) == size(x, 1)
    mul!(x, sqrt(a.inv_value))
end

function unwhiten(a::ScalMat, x::VecOrMat{Float64})
    @check_argdims dim(a) == size(x, 1)    
    x * sqrt(a.value)  
end

function unwhiten!(a::ScalMat, x::VecOrMat{Float64})
    @check_argdims dim(a) == size(x, 1)
    mul!(x, sqrt(a.value))
end

# quadratic forms

function quad(a::ScalMat, x::Vector{Float64})
    @check_argdims dim(a) == size(x, 1)
    abs2(nrm2(x)) * a.value
end

function invquad(a::ScalMat, x::Vector{Float64})
    @check_argdims dim(a) == size(x, 1)
    abs2(nrm2(x)) * a.inv_value
end

function quad!(r::AbstractArray{Float64}, a::ScalMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 1)
    mul!(vsqsum!(r, x, 1), a.value)
end

function invquad!(r::AbstractArray{Float64}, a::ScalMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 1)
    mul!(vsqsum!(r, x, 1), a.inv_value)
end

function X_A_Xt(a::ScalMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 2)
    gemm('N', 'T', a.value, x, x)
end

function Xt_A_X(a::ScalMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 1)
    gemm('T', 'N', a.value, x, x)
end

function X_invA_Xt(a::ScalMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 2)
    gemm('N', 'T', a.inv_value, x, x)
end

function Xt_invA_X(a::ScalMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 1)
    gemm('T', 'N', a.inv_value, x, x)
end


#################################################
#
#	generic functions for p.d. matrices
#
#################################################

* (c::Float64, a::AbstractPDMat) = a * c
/ (a::AbstractPDMat, c::Float64) = a * inv(c)

function quad(a::AbstractPDMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 1)
    r = Array(Float64, size(x,2))
    quad!(r, a, x)
    r
end

function invquad(a::AbstractPDMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 1)
    r = Array(Float64, size(x,2))
    invquad!(r, a, x)
    r
end


#################################################
#
#   addition
#
#################################################

# addition between p.d. matrices and ordinary ones

+ (a::PDMat,    b::Matrix{Float64}) = a.mat + b
+ (a::PDiagMat, b::Matrix{Float64}) = add_diag(b, a.diag)
+ (a::ScalMat,  b::Matrix{Float64}) = add_diag(b, a.value)

+ (a::Matrix{Float64}, b::AbstractPDMat) = b + a

add!(a::Matrix{Float64}, b::PDMat) = add!(a, b.mat)
add!(a::Matrix{Float64}, b::PDiagMat) = add_diag!(a, b.diag)
add!(a::Matrix{Float64}, b::ScalMat) = add_diag!(a, b.value)

add_scal!(a::Matrix{Float64}, b::PDMat, c::Float64) = axpy!(c, b.mat, a)
add_scal!(a::Matrix{Float64}, b::PDiagMat, c::Float64) = add_diag!(a, b.diag, c)
add_scal!(a::Matrix{Float64}, b::ScalMat, c::Float64) = add_diag!(a, b.value * c)

add_scal(a::Matrix{Float64}, b::AbstractPDMat, c::Float64) = add_scal!(copy(a), b, c)

# between pdmat and pdmat

+ (a::PDMat, b::AbstractPDMat) = PDMat(a.mat + full(b))
+ (a::PDiagMat, b::AbstractPDMat) = PDMat(add_diag!(full(b), a.diag))
+ (a::ScalMat, b::AbstractPDMat) = PDMat(add_diag!(full(b), a.value))

+ (a::PDMat, b::PDMat) = PDMat(a.mat + b.mat)
+ (a::PDMat, b::PDiagMat) = PDMat(add_diag(a.mat, b.diag))
+ (a::PDMat, b::ScalMat) = PDMat(add_diag(a.mat, b.value))

+ (a::PDiagMat, b::PDMat) = PDMat(add_diag(b.mat, a.diag))
+ (a::PDiagMat, b::PDiagMat) = PDiagMat(a.diag + b.diag)
+ (a::PDiagMat, b::ScalMat) = PDiagMat(a.diag + b.value)

+ (a::ScalMat, b::PDMat) = PDMat(add_diag(b.mat, a.value))
+ (a::ScalMat, b::PDiagMat) = PDiagMat(a.value + b.diag)
+ (a::ScalMat, b::ScalMat) = ScalMat(a.dim, a.value + b.value)

add_scal(a::PDMat, b::AbstractPDMat, c::Float64) = PDMat(a.mat + full(b * c))
add_scal(a::PDiagMat, b::AbstractPDMat, c::Float64) = PDMat(add_diag!(full(b * c), a.diag))
add_scal(a::ScalMat, b::AbstractPDMat, c::Float64) = PDMat(add_diag!(full(b * c), a.value))

add_scal(a::PDMat, b::PDMat, c::Float64) = PDMat(a.mat + b.mat * c)
add_scal(a::PDMat, b::PDiagMat, c::Float64) = PDMat(add_diag(a.mat, b.diag, c))
add_scal(a::PDMat, b::ScalMat, c::Float64) = PDMat(add_diag(a.mat, b.value * c))

add_scal(a::PDiagMat, b::PDMat, c::Float64) = PDMat(add_diag!(b.mat * c, a.diag))
add_scal(a::PDiagMat, b::PDiagMat, c::Float64) = PDiagMat(a.diag + b.diag * c)
add_scal(a::PDiagMat, b::ScalMat, c::Float64) = PDiagMat(a.diag + b.value * c)

add_scal(a::ScalMat, b::PDMat, c::Float64) = PDMat(add_diag!(b.mat * c, a.value))
add_scal(a::ScalMat, b::PDiagMat, c::Float64) = PDiagMat(a.value + b.diag * c)
add_scal(a::ScalMat, b::ScalMat, c::Float64) = ScalMat(a.dim, a.value + b.value * c)


