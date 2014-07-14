
# various transforms

### Standardization

abstract Standardize

transform!{T<:FloatingPoint, S<:Standardize}(t::S, x::DenseArray{T,1}) = transform!(x, t, x)
transform!{T<:FloatingPoint, S<:Standardize}(t::S, x::DenseArray{T,2}) = transform!(x, t, x)

transform{T<:Real, S<:Standardize}(t::S, x::DenseArray{T,1}) = transform!(Array(Float64, size(x)), t, x)
transform{T<:Real, S<:Standardize}(t::S, x::DenseArray{T,2}) = transform!(Array(Float64, size(x)), t, x)

function standardize{T<:Real, S<:Standardize}(::Type{S}, X::DenseArray{T,2}; args...)
    t = estimate(S, X; args...)
    Y = transform(t, X)
    return (Y, t)
end

function standardize!{T<:Real, S<:Standardize}(::Type{S}, X::DenseArray{T,2}; args...)
    t = estimate(S, X; args...)
    Y = transform!(t, X)
    return (Y, t)
end

# z-score
immutable ZScore <: Standardize
    dim::Int
    mean::Vector{Float64}
    scale::Vector{Float64}

    function ZScore(d::Int, m::Vector{Float64}, s::Vector{Float64})
        lenm = length(m)
        lens = length(s)
        lenm == d || lenm == 0 || throw(DimensionMismatch("Inconsistent dimensions."))
        lens == d || lens == 0 || throw(DimensionMismatch("Inconsistent dimensions."))
        new(d, m, s)
    end
end

indim(t::ZScore) = t.dim
outdim(t::ZScore) = t.dim

function transform!{YT<:Real,XT<:Real}(y::DenseArray{YT,1}, t::ZScore, x::DenseArray{XT,1})
    d = t.dim
    length(x) == length(y) == d || throw(DimensionMismatch("Inconsistent dimensions."))

    m = t.mean
    s = t.scale

    if isempty(m)
        if isempty(s)
            if !is(x, y)
                copy!(y, x)
            end
        else
            for i = 1:d
                @inbounds y[i] = x[i] * s[i]
            end
        end
    else
        if isempty(s)
            for i = 1:d
                @inbounds y[i] = x[i] - m[i]
            end
        else
            for i = 1:d
                @inbounds y[i] = s[i] * (x[i] - m[i])
            end
        end
    end
    return y
end

function transform!{YT<:Real,XT<:Real}(y::DenseArray{YT,2}, t::ZScore, x::DenseArray{XT,2})
    d = t.dim
    size(x,1) == size(y,1) == d || throw(DimensionMismatch("Inconsistent dimensions."))
    n = size(x,2)
    size(y,2) == n || throw(DimensionMismatch("Inconsistent dimensions."))

    m = t.mean
    s = t.scale

    if isempty(m)
        if isempty(s)
            if !is(x, y)
                copy!(y, x)
            end
        else
            for j = 1:n
                xj = view(x, :, j)
                yj = view(y, :, j)
                for i = 1:d
                    @inbounds yj[i] = xj[i] * s[i]
                end
            end
        end
    else
        if isempty(s)
            for j = 1:n
                xj = view(x, :, j)
                yj = view(y, :, j)
                for i = 1:d
                    @inbounds yj[i] = xj[i] - m[i]
                end
            end
        else
            for j = 1:n
                xj = view(x, :, j)
                yj = view(y, :, j)
                for i = 1:d
                    @inbounds yj[i] = s[i] * (xj[i] - m[i])
                end
            end
        end
    end
    return y
end

# estimate a standardize transform
function estimate{T<:Real}(::Type{ZScore}, X::DenseArray{T,2}; args...)
    d, n = size(X)
    n >= 2 || error("X must contain at least two columns.")

    center=true
    scale=true
    for (k,v) in args
        if k == :center
            center = v
        elseif k == :scale
            scale = v
        end
    end
    m = Array(Float64, ifelse(center, d, 0))
    s = Array(Float64, ifelse(scale, d, 0))

    if center
        fill!(m, 0.0)
        for j = 1:n
            xj = view(X, :, j)
            for i = 1:d
                @inbounds m[i] += xj[i]
            end
        end
        scale!(m, 1.0 / n)
    end

    if scale
        fill!(s, 0.0)
        if center
            for j = 1:n
                xj = view(X, :, j)
                for i = 1:d
                    @inbounds s[i] += abs2(xj[i] - m[i])
                end
            end
        else
            for j = 1:n
                xj = view(X, :, j)
                for i = 1:d
                    @inbounds s[i] += abs2(xj[i])
                end
            end
        end
        for i = 1:d
            @inbounds s[i] = sqrt((n - 1) / s[i])
        end
    end

    return ZScore(d, m, s)
end

# MinMax normalization

immutable MinMax  <: Standardize
    dim::Int
    min::Vector{Float64}
    max::Vector{Float64}

    function MinMax(d::Int, min::Vector{Float64}, max::Vector{Float64})
        lenmin = length(min)
        lenmax = length(max)
        lenmin == d || lenmin == 0 || throw(DimensionMismatch("Inconsistent dimensions."))
        lenmax == d || lenmax == 0 || throw(DimensionMismatch("Inconsistent dimensions."))
        new(d, min, max)
    end
end

indim(t::MinMax) = t.dim
outdim(t::MinMax) = t.dim

function estimate{T<:Real}(::Type{MinMax}, X::DenseArray{T,2}; args...)
    d, n = size(X)
    xmin, xmax = foldl(
        (v,e)-> begin
            push!(v[1], e[1])
            push!(v[2], e[2])
            v
        end, (Float64[], Float64[]), mapslices(extrema, X, 2))
    return MinMax(d, xmin, xmax)
end

function transform!{YT<:Real,XT<:Real}(y::DenseArray{YT,1}, t::MinMax, x::DenseArray{XT,1})
    d = t.dim
    length(x) == length(y) == d || throw(DimensionMismatch("Inconsistent dimensions."))

    for i = 1:d
        @inbounds y[i] = (x[i] - t.min[i]) / t.max[i]
    end
    return y
end

function transform!{YT<:Real,XT<:Real}(y::DenseArray{YT,2}, t::MinMax, x::DenseArray{XT,2})
    d = t.dim
    size(x,1) == size(y,1) == d || throw(DimensionMismatch("Inconsistent dimensions."))
    n = size(x,2)
    size(y,2) == n || throw(DimensionMismatch("Inconsistent dimensions."))

    for j = 1:n
        xj = view(x, :, j)
        yj = view(y, :, j)
        for i = 1:d
            @inbounds yj[i] = (xj[i] - t.min[i]) / t.max[i]
        end
    end
    return y
end