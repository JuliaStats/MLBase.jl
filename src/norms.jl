# Norm computation and normalization

function weighted_sqnorm{T<:Number}(x::AbstractVector{T}, w::AbstractVector{T})
    s = 0.
    n = length(x)
    @check_argdims n == length(w)
    for i = 1 : n
        s += w[i] * abs2(x[i])
    end
    s
end

function colwise_dot!{S<:Number, T<:Number}(r::AbstractArray{S}, x::Matrix{T}, y::Matrix{T})
    m = size(x, 1)
    n = size(x, 2)
    @check_argdims size(y, 1) == m && size(y, 2) == n && length(r) == n
    b::Int = 0
    for j = 1 : n
        s = zero(S)
        for i = 1 : m            
            s += x[b + i] * y[b + i]
        end
        r[j] = s
        b += m
    end
    r
end

function colwise_dot{T<:Number}(x::Matrix{T}, y::Matrix{T})
    colwise_dot!(Array(Float64, size(x,2)), x, y)    
end

function colwise_sqnorm!{S<:Number, T<:Number}(r::AbstractArray{S}, x::Matrix{T})
    m = size(x, 1)
    n = size(x, 2)
    @check_argdims length(r) == n
    b::Int = 0
    for j = 1 : n
        s = zero(S)
        for i = 1 : m 
            s += abs2(x[b + i])
        end
        r[j] = s
        b += m
    end
    r
end

function colwise_sqnorm{T<:Number}(x::Matrix{T})
    colwise_sqnorm!(Array(Float64, size(x,2)), x)    
end

function colwise_weighted_sqnorm!{S<:Number, T<:Number}(r::AbstractArray{S}, x::Matrix{T}, w::Vector{T})
    m = size(x, 1)
    n = size(x, 2)
    @check_argdims length(w) == m && length(r) == n
    b::Int = 0
    for j = 1 : n
        s = zero(S)
        for i = 1 : m 
            s += abs2(x[b + i]) * w[i]
        end
        r[j] = s
        b += m
    end
    r
end

function colwise_weighted_sqnorm{T<:Number}(x::Matrix{T}, w::Vector{T})
    colwise_weighted_sqnorm!(Array(Float64, size(x,2)), x, w)    
end