# Basic computation routines

macro check_argdims(cond)
    :( if !($(esc(cond)))
        throw(ArgumentError("Invalid argument dimensions.")) 
    end)  
end

#################################################
#
#  Basic arithmetics
#
#################################################

# add

function add!{T<:Number}(a::Array{T}, b::Array{T})
    @check_argdims length(a) == length(b)
    for i = 1 : length(a)
        a[i] += b[i]
    end
    a
end

function add!{T<:Number}(a::Array{T}, b::T)
    for i = 1 : length(a)
        a[i] += b
    end
    a
end

function add_cols!{T <: Number}(a::Matrix{T}, b::Vector{T})
    @check_argdims size(a, 1) == length(b)
    for j = 1 : size(a, 2)
        for i = 1 : size(a, 1)
            a[i,j] += b[i]
        end
    end
    a
end

function add_cols{T <: Number}(a::Matrix{T}, b::Vector{T})
    @check_argdims size(a, 1) == length(b)
    r = similar(a)
    for j = 1 : size(a, 2)
        for i = 1 : size(a, 1)
            r[i,j] = a[i,j] + b[i]
        end
    end
    r
end

function add_rows!{T <: Number}(a::Matrix{T}, b::Vector{T})
    @check_argdims size(a, 2) == length(b)
    for j = 1 : size(a, 2)
        bj = b[j]
        for i = 1 : size(a, 1)
            a[i,j] += bj
        end
    end
    a
end

function add_rows{T <: Number}(a::Matrix{T}, b::Vector{T})
    @check_argdims size(a, 2) == length(b)
    r = similar(a)
    for j = 1 : size(a, 2)
        bj = b[j]
        for i = 1 : size(a, 1)
            r[i,j] = a[i,j] + bj
        end
    end
    r
end

# sub

function sub!{T<:Number}(a::Array{T}, b::Array{T})
    @check_argdims length(a) == length(b)
    for i = 1 : length(a)
        a[i] -= b[i]
    end
    a
end

function sub!{T<:Number}(a::Array{T}, b::T)
    for i = 1 : length(a)
        a[i] -= b
    end
    a
end

function sub_cols!{T <: Number}(a::Matrix{T}, b::Vector{T})
    @check_argdims size(a, 1) == length(b)
    for j = 1 : size(a, 2)
        for i = 1 : size(a, 1)
            a[i,j] -= b[i]
        end
    end
    a
end

function sub_cols{T <: Number}(a::Matrix{T}, b::Vector{T})
    @check_argdims size(a, 1) == length(b)
    r = similar(a)
    for j = 1 : size(a, 2)
        for i = 1 : size(a, 1)
            r[i,j] = a[i,j] - b[i]
        end
    end
    r
end

function sub_rows!{T <: Number}(a::Matrix{T}, b::Vector{T})
    @check_argdims size(a, 2) == length(b)
    for j = 1 : size(a, 2)
        bj = b[j]
        for i = 1 : size(a, 1)
            a[i,j] -= bj
        end
    end
    a
end

function sub_rows{T <: Number}(a::Matrix{T}, b::Vector{T})
    @check_argdims size(a, 2) == length(b)
    r = similar(a)
    for j = 1 : size(a, 2)
        bj = b[j]
        for i = 1 : size(a, 1)
            r[i,j] = a[i,j] - bj
        end
    end
    r
end

# mul

function mul!{T<:Number}(a::Array{T}, b::Array{T})
    @check_argdims length(a) == length(b)
    for i = 1 : length(a)
        a[i] *= b[i]
    end
    a
end

function mul!{T<:Number}(a::Array{T}, b::T)
    for i = 1 : length(a)
        a[i] *= b
    end
    a
end

function mul_cols!{T <: Number}(a::Matrix{T}, b::Vector{T})
    @check_argdims size(a, 1) == length(b)
    for j = 1 : size(a, 2)
        for i = 1 : size(a, 1)
            a[i,j] *= b[i]
        end
    end
    a
end

function mul_cols{T <: Number}(a::Matrix{T}, b::Vector{T})
    @check_argdims size(a, 1) == length(b)
    r = similar(a)
    for j = 1 : size(a, 2)
        for i = 1 : size(a, 1)
            r[i,j] = a[i,j] * b[i]
        end
    end
    r
end

function mul_rows!{T <: Number}(a::Matrix{T}, b::Vector{T})
    @check_argdims size(a, 2) == length(b)
    for j = 1 : size(a, 2)
        bj = b[j]
        for i = 1 : size(a, 1)
            a[i,j] *= bj
        end
    end
    a
end

function mul_rows{T <: Number}(a::Matrix{T}, b::Vector{T})
    @check_argdims size(a, 2) == length(b)
    r = similar(a)
    for j = 1 : size(a, 2)
        bj = b[j]
        for i = 1 : size(a, 1)
            r[i,j] = a[i,j] * bj
        end
    end
    r
end


#################################################
#
#  dot and norms
#
#################################################

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

