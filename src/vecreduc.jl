# Vector statistics


function _reduc_length(m::Int, n::Int, dim::Integer)
    dim == 1 ? n :
    dim == 2 ? m :
    throw(ArgumentError("The value of dim must be either 1 or 2."))
end

#################################################
#
#  Generic vector reduction
#
#################################################

abstract AbstractReduction

# one argument

function vreduce!(r::Array, op::AbstractReduction, x::Matrix, dim::Integer)
    m = size(x, 1)
    n = size(x, 2)
    @check_argdims length(r) == _reduc_length(m, n, dim)

    if dim == 1
        if m == 0
            fill!(r, empty_value(op, eltype(x)))
        elseif m == 1
            for j = 1 : n
                r[j] = init_value(op, x[j])
            end
        else            
            for j in 1 : n
                rj = init_value(op, x[1,j])
                for i in 2 : m
                    rj = combine_value(op, rj, x[i,j])
                end
                r[j] = rj
            end
        end

    else # _reduc_length ensures: dim == 2
        if n == 0
            fill!(r, empty_value(op, eltype(x)))
        elseif n == 1
            for i in 1 : m
                r[i] = init_value(op, x[i])
            end
        else
            for i in 1 : m
                r[i] = init_value(op, x[i,1])
            end
            for j in 2 : n
                for i in 1 : m
                    r[i] = combine_value(op, r[i], x[i,j])
                end
            end
        end
    end
    r
end

function vreduce(op::AbstractReduction, x::Matrix, dim::Integer)
    rlen::Int = _reduc_length(size(x, 1), size(x, 2), dim)
    r = Array(result_type(op, eltype(x)), rlen)
    vreduce!(r, op, x, dim)
end

# two argument

function vreduce!(r::Array, op::AbstractReduction, x::Matrix, y::Matrix, dim::Integer)
    m = size(x, 1)
    n = size(x, 2)
    @check_argdims size(y, 1) == m && size(y, 2) == n 
    @check_argdims length(r) == _reduc_length(m, n, dim)

    if dim == 1
        if m == 0
            fill!(r, empty_value(op, eltype(x), eltype(y)))
        elseif m == 1
            for j = 1 : n
                r[j] = init_value(op, x[j], y[j])
            end
        else            
            for j in 1 : n
                rj = init_value(op, x[1,j], y[1,j])
                for i in 2 : m
                    rj = combine_value(op, rj, x[i,j], y[i,j])
                end
                r[j] = rj
            end
        end

    else # _reduc_length ensures: dim == 2
        if n == 0
            fill!(r, empty_value(op, eltype(x), eltype(y)))
        elseif n == 1
            for i in 1 : m
                r[i] = init_value(op, x[i], y[i])
            end
        else
            for i in 1 : m
                r[i] = init_value(op, x[i,1], y[i,1])
            end
            for j in 2 : n
                for i in 1 : m
                    r[i] = combine_value(op, r[i], x[i,j], y[i,j])
                end
            end
        end
    end
    r
end

function vreduce(op::AbstractReduction, x::Matrix, y::Matrix, dim::Integer)
    rlen::Int = _reduc_length(size(x, 1), size(x, 2), dim)
    r = Array(result_type(op, eltype(x), eltype(y)), rlen)
    vreduce!(r, op, x, y, dim)
end


#################################################
#
#  One-argument reductors
#
#################################################

type SumReduc <: AbstractReduction end

result_type{T<:Number}(op::SumReduc, ty::Type{T}) = T
empty_value{T<:Number}(op::SumReduc, ty::Type{T}) = zero(T)
init_value{T<:Number}(op::SumReduc, x::T) = x
combine_value{T<:Number}(op::SumReduc, s::T, x::T) = s + x

vsum!(r::Array, x::Matrix, dim::Int) = vreduce!(r, SumReduc(), x, dim)
vsum(x::Matrix, dim::Int) = vreduce(SumReduc(), x, dim)

function vmean!{T<:Number}(r::Array{T}, x::Matrix, dim::Int)
    m = size(x, dim);
    if m > 0
        vsum!(r, x, dim)
        mul!(r, one(T) / m)
    else
        fill!(r, nan(T))
    end
    r
end

function vmean{T<:Number}(x::Matrix{T}, dim::Int)
    rlen::Int = _reduc_length(size(x, 1), size(x, 2), dim)
    vmean!(Array(T, rlen), x, dim)
end


type MaxReduc <: AbstractReduction end

result_type{T<:Real}(op::MaxReduc, ty::Type{T}) = T
empty_value{T<:Real}(op::MaxReduc, ty::Type{T}) = typemin(T)
init_value{T<:Real}(op::MaxReduc, x::T) = x
combine_value{T<:Real}(op::MaxReduc, s::T, x::T) = s > x ? s : x

vmax!(r::Array, x::Matrix, dim::Int) = vreduce!(r, MaxReduc(), x, dim)
vmax(x::Matrix, dim::Int) = vreduce(MaxReduc(), x, dim)


type MinReduc <: AbstractReduction end

result_type{T<:Real}(op::MinReduc, ty::Type{T}) = T
empty_value{T<:Real}(op::MinReduc, ty::Type{T}) = typemax(T)
init_value{T<:Real}(op::MinReduc, x::T) = x
combine_value{T<:Real}(op::MinReduc, s::T, x::T) = s < x ? s : x

vmin!(r::Array, x::Matrix, dim::Int) = vreduce!(r, MinReduc(), x, dim)
vmin(x::Matrix, dim::Int) = vreduce(MinReduc(), x, dim)


type AsumReduc <: AbstractReduction end

result_type{T<:Number}(op::AsumReduc, ty::Type{T}) = T
empty_value{T<:Number}(op::AsumReduc, ty::Type{T}) = zero(T)
init_value{T<:Number}(op::AsumReduc, x::T) = abs(x)
combine_value{T<:Number}(op::AsumReduc, s::T, x::T) = s + abs(x)

vasum!(r::Array, x::Matrix, dim::Int) = vreduce!(r, AsumReduc(), x, dim)
vasum(x::Matrix, dim::Int) = vreduce(AsumReduc(), x, dim)


type AmaxReduc <: AbstractReduction end

result_type{T<:Number}(op::AmaxReduc, ty::Type{T}) = T
empty_value{T<:Number}(op::AmaxReduc, ty::Type{T}) = zero(T)
init_value{T<:Number}(op::AmaxReduc, x::T) = abs(x)
combine_value{T<:Number}(op::AmaxReduc, s::T, x::T) = (a = abs(x); s > a ? s : a)

vamax!(r::Array, x::Matrix, dim::Int) = vreduce!(r, AmaxReduc(), x, dim)
vamax(x::Matrix, dim::Int) = vreduce(AmaxReduc(), x, dim)


type AminReduc <: AbstractReduction end

result_type{T<:Number}(op::AminReduc, ty::Type{T}) = T
empty_value{T<:Number}(op::AminReduc, ty::Type{T}) = typemax(T)
init_value{T<:Number}(op::AminReduc, x::T) = abs(x)
combine_value{T<:Number}(op::AminReduc, s::T, x::T) = (a = abs(x); s < a ? s : a)

vamin!(r::Array, x::Matrix, dim::Int) = vreduce!(r, AminReduc(), x, dim)
vamin(x::Matrix, dim::Int) = vreduce(AminReduc(), x, dim)


type SqsumReduc <: AbstractReduction end

result_type{T<:Number}(op::SqsumReduc, ty::Type{T}) = T
empty_value{T<:Number}(op::SqsumReduc, ty::Type{T}) = zero(T)
init_value{T<:Number}(op::SqsumReduc, x::T) = abs2(x)
combine_value{T<:Number}(op::SqsumReduc, s::T, x::T) = s + abs2(x)

vsqsum!(r::Array, x::Matrix, dim::Int) = vreduce!(r, SqsumReduc(), x, dim)
vsqsum(x::Matrix, dim::Int) = vreduce(SqsumReduc(), x, dim)

function weighted_sqsum(x::Vector, w::Vector)
    @check_argdims length(x) == length(w)
    s = 0.
    for i in 1 : length(x)
        s += abs2(x[i]) * w[i]
    end
    s
end


immutable PowsumReduc{T<:Real} <: AbstractReduction 
    p::T
end

result_type{T<:Number}(op::PowsumReduc, ty::Type{T}) = T
empty_value{T<:Number}(op::PowsumReduc, ty::Type{T}) = zero(T)
init_value{T<:Number}(op::PowsumReduc, x::T) = abs(x) ^ op.p
combine_value{T<:Number}(op::PowsumReduc, s::T, x::T) = s + abs(x) ^ op.p

vpowsum!(r::Array, x::Matrix, p::Real, dim::Int) = vreduce!(r, PowsumReduc(p), x, dim)
vpowsum(x::Matrix, p::Real, dim::Int) = vreduce(PowsumReduc(p), x, dim)


#################################################
#
#  Two-argument reductors
#
#################################################

type DotReduc <: AbstractReduction end

result_type{T<:Number}(op::DotReduc, ty1::Type{T}, ty2::Type{T}) = T
empty_value{T<:Number}(op::DotReduc, ty1::Type{T}, ty2::Type{T}) = zero(T)
init_value{T<:Number}(op::DotReduc, x::T, y::T) = x * y
combine_value{T<:Number}(op::DotReduc, s::T, x::T, y::T) = s + x * y

vdot!(r::Array, x::Matrix, y::Matrix, dim::Int) = vreduce!(r, DotReduc(), x, y, dim)
vdot(x::Matrix, y::Matrix, dim::Int) = vreduce(DotReduc(), x, y, dim)


type SqdiffsumReduc <: AbstractReduction end

result_type{T<:Number}(op::SqdiffsumReduc, ty1::Type{T}, ty2::Type{T}) = T
empty_value{T<:Number}(op::SqdiffsumReduc, ty1::Type{T}, ty2::Type{T}) = zero(T)
init_value{T<:Number}(op::SqdiffsumReduc, x::T, y::T) = abs2(x - y)
combine_value{T<:Number}(op::SqdiffsumReduc, s::T, x::T, y::T) = s + abs2(x - y)

vsqdiffsum!(r::Array, x::Matrix, y::Matrix, dim::Int) = vreduce!(r, SqdiffsumReduc(), x, y, dim)
vsqdiffsum(x::Matrix, y::Matrix, dim::Int) = vreduce(SqdiffsumReduc(), x, y, dim)


