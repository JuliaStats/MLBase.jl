# Utilities

## add/subtract a vector

function addvec!(x::StridedVector, v::StridedVector)
    n = length(x)
    length(v) == n || error("Inconsistent vector length.")
    @inbounds for i = 1:n
        x[i] += v[i]
    end
    return x
end

function addvec!(x::StridedVector, v::StridedVector, c::Real)
    n = length(x)
    length(v) == n || error("Inconsistent vector length.")
    @inbounds for i = 1:n
        x[i] += v[i] * c
    end
    return x
end

function subtractvec!(x::StridedVector, v::StridedVector)
    n = length(x)
    length(v) == n || error("Inconsistent vector length.")
    @inbounds for i = 1:n
        x[i] -= v[i]
    end
    return x
end

function subtractvec!(x::StridedVector, v::StridedVector, c::Real)
    n = length(x)
    length(v) == n || error("Inconsistent vector length.")
    @inbounds for i = 1:n
        x[i] -= v[i] * c
    end
    return x
end

## add/subtract each column

function addeachcol!(x::DenseMatrix, v::StridedVector)
    n = length(x)
    length(v) == n || error("Inconsistent vector length.")
    @inbounds for i = 1:n
        x[i] += v[i]
    end
    return x
end

function addeachcol!(x::StridedVector, v::StridedVector, c::Real)
    n = length(x)
    length(v) == n || error("Inconsistent vector length.")
    @inbounds for i = 1:n
        x[i] += v[i] * c
    end
    return x
end

function subtracteachcol!(x::StridedVector, v::StridedVector)
    n = length(x)
    length(v) == n || error("Inconsistent vector length.")
    @inbounds for i = 1:n
        x[i] -= v[i]
    end
    return x
end

function subtracteachcol!(x::StridedVector, v::StridedVector, c::Real)
    n = length(x)
    length(v) == n || error("Inconsistent vector length.")
    @inbounds for i = 1:n
        x[i] -= v[i] * c
    end
    return x
end







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

function repeach{T}(x::AbstractVector{T}, ns::IntegerVector)
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

function repeachcol{T}(x::DenseArray{T,2}, ns::IntegerVector)
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

function repeachrow{T}(x::DenseArray{T,2}, ns::IntegerVector)
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

## count the number of equal/non-equal pairs

function counteq(a::IntegerVector, b::IntegerVector)
    n = length(a)
    length(b) == n || throw(DimensionMismatch("Inconsistent lengths."))
    c = 0
    for i = 1:n
        if a[i] == b[i]
            c += 1
        end
    end
    return c
end

function countne(a::IntegerVector, b::IntegerVector)
    n = length(a)
    length(b) == n || throw(DimensionMismatch("Inconsistent lengths."))
    c = 0
    for i = 1:n
        if a[i] != b[i]
            c += 1
        end
    end
    return c
end

