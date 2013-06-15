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


#################################################
#
#  Specific reductors
#
#################################################

type SumReduc <: AbstractReduction end

result_type{T<:Number}(op::SumReduc, ty::Type{T}) = T
empty_value{T<:Number}(op::SumReduc, ty::Type{T}) = zero(T)
init_value{T<:Number}(op::SumReduc, x::T) = x
combine_value{T<:Number}(op::SumReduc, s::T, x::T) = s + x

vsum!(r::Array, x::Matrix, dim::Int) = vreduce!(r, SumReduc(), x, dim)
vsum(x::Matrix, dim::Int) = vreduce(SumReduc(), x, dim)


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



