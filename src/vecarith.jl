# Basic vector arithmetics

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

# add_diag

function add_diag!(a::Matrix, v::Number)
    d::Int = min(size(a, 1), size(a, 2))
    for i in 1 : d
        a[i,i] += v
    end
    a
end

function add_diag!(a::Matrix, v::Vector)
    d::Int = min(size(a, 1), size(a, 2))
    @check_argdims d == length(v)
    for i in 1 : d
        a[i,i] += v[i]
    end
    a
end

function add_diag!(a::Matrix, v::Vector, c::Number)
    d::Int = min(size(a, 1), size(a, 2))
    @check_argdims d == length(v)
    for i in 1 : d
        a[i,i] += v[i] * c
    end    
    a
end

add_diag(a::Matrix, v::Number) = add_diag!(copy(a), v)
add_diag(a::Matrix, v::Vector) = add_diag!(copy(a), v)
add_diag(a::Matrix, v::Vector, c::Number) = add_diag!(copy(a), v, c)





