# Utilities

## repeat each element in a vector

function repeach(x::AbstractVector{T}, n::Integer) where T
    k = length(x)
    r = Array{T}(undef, k * n)
    p = 0
    @inbounds for i = 1:k
        xi = x[i]
        for j = 1:n
            r[p += 1] = xi
        end
    end
    return r
end

function repeach(x::AbstractVector{T}, ns::IntegerVector) where T
    k = length(x)
    length(ns) == k || throw(DimensionMismatch("length(ns) should be equal to k."))
    r = Array{T}(undef, sum(ns))
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

function repeachcol(x::DenseArray{T,2}, n::Integer) where T
    m = size(x, 1)
    k = size(x, 2)
    r = Array{T}(undef, m, k * n)
    p = 0
    @inbounds for i = 1:k
        xi = view(x, :, i)
        for j = 1:n
            r[:, p += 1] = xi
        end
    end
    return r
end

function repeachcol(x::DenseArray{T,2}, ns::IntegerVector) where T
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

function repeachrow(x::DenseArray{T,2}, n::Integer) where T
    k = size(x, 1)
    m = size(x, 2)
    r = Array{T}(undef, k * n, m)
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

function repeachrow(x::DenseArray{T,2}, ns::IntegerVector) where T
    k = size(x, 1)
    m = size(x, 2)
    r = Array{T}(undef, sum(ns), m)
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

## return the unique values of A and a vector of vectors of indexes to reconstruct
## the original array

function unique_inverse(A::AbstractArray)
    out = Array{eltype(A)}(undef, 0)
    out_idx = Array{Vector{Int}}(undef, 0)
    seen = Dict{eltype(A), Int}()
    for (idx, x) in enumerate(A)
        if !in(x, keys(seen))
            seen[x] = length(seen) + 1
            push!(out, x)
            push!(out_idx, Int[])
        end
        push!(out_idx[seen[x]], idx)
    end
    out, out_idx
end
