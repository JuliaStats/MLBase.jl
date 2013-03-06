# Tools for probability computation

using Devectorize

# compute entropy of discrete distribution

function entropy(p::AbstractVector{Float64})
    v = 0.
    for i = 1 : length(p)
        pi = p[i]
        if pi > 0
            v -= pi * log(pi)
        end
    end
    v
end

function entropy!(r::AbstractArray{Float64}, p::AbstractMatrix{Float64}, dim::Int)
    m, n = size(p)
    if dim == 1
        for j = 1 : n
            v = 0.
            for i = 1 : m
                pi = p[i, j]
                if pi > 0
                    v -= pi * log(pi)
                end
            end
            r[j] = v
        end

    elseif dim == 2
        fill!(r, 0.)
        for j = 1 : n
            for i = 1 : m
                pi = p[i,j]
                if pi > 0
                    r[i] -= pi * log(pi)
                end
            end
        end
    else
        throw(ArgumentError("dim must be either 1 or 2."))
    end
end

function entropy(p::AbstractMatrix{Float64}, dim::Int)
    if dim == 1
        r = Array(Float64, 1, size(p, 2))
        entropy!(r, p, dim)
    elseif dim == 2
        r = Array(Float64, size(p, 1), 1)
        entropy!(r, p, dim)
    else
        throw(ArgumentError("dim must be either 1 or 2."))
    end
    return r
end


# a numerically stable method to compute
#
#   log( sum_i exp(x_i) )
#
function logsumexp(x::AbstractVector{Float64})
    mx = max(x)
    n = length(x)
    s = 0.
    for i = 1 : n
        s += exp(x[i] - mx)
    end
    log(s) + mx
end

function logsumexp!(r::AbstractArray{Float64}, x::AbstractMatrix{Float64}, dim::Int)
    if dim == 1
        if length(r) != size(x, 2)
            throw(ArgumentError("The length or must match the number of columns in x."))
        end
        m, n = size(x)

        @devec r[:] = max(x, (), 1)
        for j = 1 : n
            s = 0.
            mx = r[j]
            for i = 1 : m
                s += exp(x[i,j] - mx)
            end
            r[j] = log(s) + mx
        end

    elseif dim == 2
        if length(r) != size(x, 1)
            throw(ArgumentError("The length or must match the number of rows in x."))
        end
        m, n = size(x)

        @devec r[:] = max(x, (), 2)
        s = zeros(m)
        for j = 1 : n
            for i = 1 : m
                s[i] += exp(x[i,j] - r[i])
            end
        end
        @devec r[:] = r + log(s)

    else
        throw(ArgumentError("dim must be either 1 or 2."))
    end
end


function logsumexp(x::AbstractMatrix{Float64}, dim::Int)
    if dim == 1
        r = zeros(1, size(x, 2))
        logsumexp!(r, x, dim)
    elseif dim == 2
        r = zeros(size(x, 1), 1)
        logsumexp!(r, x, dim)
    else
        throw(ArgumentError("dim must be either 1 or 2."))
    end
    return r
end



# numerical stable method to compute softmax
#
#   r[i] = exp(x[i]) / sum(exp(x))
#

function softmax!(r::AbstractVector{Float64}, x::AbstractVector{Float64})
    if length(r) != length(x)
        throw(ArgumentError("The lengths of r and x must match."))
    end
    n = length(x)
    mx = max(x)

    # must use double as accumulator, even x is single
    # otherwise, errors can build up very fast
    s = 0.0
    for i = 1 : n
        r[i] = exp(x[i] - mx)
        s += r[i]
    end
    inv_s = 1/s

    @devec r[:] .*= inv_s
end

function softmax(x::AbstractVector{Float64})
    r = similar(x)
    softmax!(r, x)
    return r
end

function softmax!(r::AbstractMatrix{Float64}, x::AbstractMatrix{Float64}, dim::Int)
    if !(dim == 1 || dim == 2)
        throw(ArgumentError("dim must be either 1 or 2."))
    end
    if size(r) != size(x)
        throw(ArgumentError("The sizes of r and x must match."))
    end
    m, n = size(x)

    if dim == 1 # by columns
        @devec mx = max(x, (), 1)
        for j = 1 : n
            s = 0.0
            for i = 1 : m
                s += (r[i,j] = exp(x[i,j] - mx[j]))
            end
            inv_s = rcp(s)
            @devec r[:,j] .*= inv_s
        end
    else
        # to make it cache-friendly, the structure is different
        @devec mx = max(x, (), 2)
        s = zeros(m)
        for j = 1 : n
            for i = 1 : m
                s[i] += (r[i,j] = exp(x[i,j] - mx[i]))
            end
        end
        @devec inv_s = rcp(s)

        for j = 1 : n
            @devec r[:,j] .*= inv_s
        end
    end
end


function softmax(x::AbstractMatrix{Float64}, dim::Int)
    r = similar(x)
    softmax!(r, x, dim)
    return r
end

