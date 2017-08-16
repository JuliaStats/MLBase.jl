
export Standardize

export standardize, standardize!, transform

# various transforms

### Standardization

struct Standardize
    dim::Int
    mean::Vector{Float64}
    scale::Vector{Float64}

    function Standardize(d::Int, m::Vector{Float64}, s::Vector{Float64})
        Base.depwarn("Standardize is deprecated. Please use mean_and_std and zscore in StatsBase instead.", :Standardize)
        lenm = length(m)
        lens = length(s)
        lenm == d || lenm == 0 || throw(DimensionMismatch("Inconsistent dimensions."))
        lens == d || lens == 0 || throw(DimensionMismatch("Inconsistent dimensions."))
        new(d, m, s)
    end
end

indim(t::Standardize) = t.dim
outdim(t::Standardize) = t.dim

function transform!(y::DenseArray{YT,1}, t::Standardize, x::DenseArray{XT,1}) where {YT<:Real,XT<:Real}
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

function transform!(y::DenseArray{YT,2}, t::Standardize, x::DenseArray{XT,2}) where {YT<:Real,XT<:Real}
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

transform!(t::Standardize, x::DenseArray{T,1}) where {T<:AbstractFloat} = transform!(x, t, x)
transform!(t::Standardize, x::DenseArray{T,2}) where {T<:AbstractFloat} = transform!(x, t, x)

transform(t::Standardize, x::DenseArray{T,1}) where {T<:Real} = transform!(Array{Float64}(size(x)), t, x)
transform(t::Standardize, x::DenseArray{T,2}) where {T<:Real} = transform!(Array{Float64}(size(x)), t, x)

# estimate a standardize transform

function estimate(::Type{Standardize}, X::DenseArray{T,2}; center::Bool=true, scale::Bool=true) where T<:Real
    d, n = size(X)
    n >= 2 || error("X must contain at least two columns.")

    m = Array{Float64}(ifelse(center, d, 0))
    s = Array{Float64}(ifelse(scale, d, 0))

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

    return Standardize(d, m, s)
end

# standardize

function standardize(X::DenseArray{T,2}; center::Bool=true, scale::Bool=true) where T<:Real
    t = estimate(Standardize, X; center=center, scale=scale)
    Y = transform(t, X)
    return (Y, t)
end

function standardize!(X::DenseArray{T,2}; center::Bool=true, scale::Bool=true) where T<:AbstractFloat
    t = estimate(Standardize, X; center=center, scale=scale)
    Y = transform!(t, X)
    return (Y, t)
end
