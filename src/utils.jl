# Utilities

## repeat each element in a vector

function repeach{T}(x::AbstractVector{T}, n::Integer)
    k = length(x)
    r = Array(T, k * n)
    p = 0
    @inbounds for i = 1:k
        xi = x[i]
        for j = 1:n
            r[p += 1] = xi 
        end
    end
    return r
end

function repeach{T,I<:Integer}(x::AbstractVector{T}, ns::AbstractVector{I})
    k = length(x)
    length(ns) == k || throw(DimensionMismatch("length(ns) should be equal to k."))
    r = Array(T, sum(ns))
    p = 0
    @inbounds for i = 1:k
        xi = x[i]
        ni = ns[i]
        for j = 1:ni
            r[p += 1] = xi
        end
    end
    return r
end

## repeat each column in a matrix

function repeachcol{T}(x::DenseArray{T,2}, n::Integer)
    m = size(x, 1)
    k = size(x, 2)
    r = Array(T, m, k * n)
    p = 0
    @inbounds for i = 1:k
        xi = view(x, :, i)
        for j = 1:n
            r[:, p += 1] = xi
        end
    end
    return r
end

function repeachcol{T,I<:Integer}(x::DenseArray{T,2}, ns::AbstractVector{I})
    m = size(x, 1)
    k = size(x, 2)
    r = zeros(T, m, sum(ns))
    p = 0
    @inbounds for i = 1:k
        xi = view(x, :, i)
        ni = ns[i]
        for j = 1:ni
            r[:, p += 1] = xi
        end
    end
    return r
end

## repeat each row in a matrix

function repeachrow{T}(x::DenseArray{T,2}, n::Integer)
    k = size(x, 1)
    m = size(x, 2)
    r = Array(T, k * n, m)
    p = 0
    @inbounds for icol = 1:m
        p = 0
        for i = 1:k
            xi = x[i, icol]
            for j = 1:n
                r[p += 1, icol] = xi
            end
        end
    end
    return r
end

function repeachrow{T,I<:Integer}(x::DenseArray{T,2}, ns::AbstractVector{I})
    k = size(x, 1)
    m = size(x, 2)
    r = Array(T, sum(ns), m)
    @inbounds for icol = 1:m
        p = 0
        for i = 1:k
            xi = x[i, icol]
            ni = ns[i]
            for j = 1:ni
                r[p += 1, icol] = xi
            end
        end
    end
    return r
end

