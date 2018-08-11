# manipulation of class labels

## simple classification

# classify

function classify(x::RealVector, ord::Ordering)
    n = length(x)
    v = x[1]
    k::Int = 1
    for i = 2:n
        @inbounds xi = x[i]
        if lt(ord, v, xi)
            v = xi
            k = i
        end
    end
    return k
end

classify(x::RealVector) = classify(x, Forward)

function classify!(r::IntegerVector, x::RealMatrix, ord::Ordering)
    m = size(x, 1)
    n = size(x, 2)
    length(r) == n || throw(DimensionMismatch("Mismatched length of r."))
    for j = 1:n
        @inbounds r[j] = classify(view(x,:,j), ord)
    end
    return r
end

classify!(r::IntegerVector, x::RealMatrix) = classify!(r, x, Forward)

# - this one throws a deprecation
classify(x::RealMatrix, ord::Ordering) = classify!(Array{Int}(undef, size(x,2)), x, ord)
classify(x::RealMatrix) = classify(x, Forward)

# classify with score(s)

function classify_withscore(x::RealVector, ord::Ordering)
    n = length(x)
    v = x[1]
    k::Int = 1
    for i = 2:n
        @inbounds xi = x[i]
        if lt(ord, v, xi)
            v = xi
            k = i
        end
    end
    return (k, v)
end

classify_withscore(x::RealVector) = classify_withscore(x, Forward)

function classify_withscores!(r::IntegerVector, s::RealVector, x::RealMatrix, ord::Ordering)
    m = size(x, 1)
    n = size(x, 2)
    length(r) == n || throw(DimensionMismatch("Mismatched length of r."))
    for j = 1:n
        k, v = classify_withscore(view(x,:,j), ord)
        @inbounds r[j] = k
        @inbounds s[j] = v
    end
    return (r, s)
end

classify_withscores!(r::IntegerVector, s::RealVector, x::RealMatrix) =
    classify_withscores!(r, s, x, Forward)

function classify_withscores(x::RealMatrix{T}, ord::Ordering) where T<:Real
    n = size(x, 2)
    r = Array{Int}(undef, n)
    s = Array{T}(undef, n)
    return classify_withscores!(r, s, x, ord)
end

classify_withscores(x::RealMatrix{T}) where {T<:Real} = classify_withscores(x, Forward)


# classify with threshold

classify(x::RealVector, t::Real, ord::Ordering) =
    ((k, v) = classify_withscore(x, ord); ifelse(lt(ord, v, t), 0, k))

classify(x::RealVector, t::Real) = classify(x, t, Forward)

function classify!(r::IntegerVector, x::RealMatrix, t::Real, ord::Ordering)
    m = size(x, 1)
    n = size(x, 2)
    length(r) == n || throw(DimensionMismatch("Mismatched length of r."))
    for j = 1:n
        @inbounds r[j] = classify(view(x,:,j), t, ord)
    end
    return r
end

classify!(r::IntegerVector, x::RealMatrix, t::Real) = classify!(r, x, t, Forward)

classify(x::RealMatrix, t::Real, ord::Ordering) = classify!(Array{Int}(undef, size(x,2)), x, t, ord)
classify(x::RealMatrix, t::Real) = classify(x, t, Forward)


## label map

struct LabelMap{K}
    vs::Vector{K}
    v2i::Dict{K,Int}

    function LabelMap{K}(vs, v2i) where K
        length(vs) == length(v2i) || throw(DimensionMismatch("lengths of vs and v2i mismatch"))
        new(vs,v2i)
    end
end

LabelMap(vs::Vector{K}, v2i::Dict{K,Int}) where {K}= LabelMap{K}(vs, v2i)

length(lmap::LabelMap) = length(lmap.vs)
keys(lmap::LabelMap) = lmap.vs

function show(io::IO, lmap::LabelMap)
    n = length(lmap)
    println(io, "LabelMap (with $n labels):")
    for (i, v) in enumerate(keys(lmap))
        println(io, "[$i] $v")
    end
end


# build a label map (value -> label) from a sequences of values
function labelmap(xs::AbstractArray{T}) where T
    l = 0
    vs = T[]
    v2i = Dict{T, Int}()
    for x in xs
        if !haskey(v2i, x)
            push!(vs, x)
            v2i[x] = (l += 1)
        end
    end
    return LabelMap(vs, v2i)
end

# use a map to encode discrete values into labels
labelencode(lmap::LabelMap{T}, x) where {T} = lmap.v2i[convert(T, x)]
labelencode(lmap::LabelMap{T}, xs::AbstractArray{T}) where {T} =
    reshape(Int[labelencode(lmap, x) for x in xs], size(xs))

# decode the label to the associated discrete value
labeldecode(lmap::LabelMap{T}, y::Int) where {T} = lmap.vs[y]
labeldecode(lmap::LabelMap{T}, ys::AbstractArray{Int}) where {T} =
    reshape(T[labeldecode(lmap, y) for y in ys], size(ys))

## group labels

function groupindices(k::Int, xs::IntegerVector; warning::Bool=true)
    gs = Array{Vector{Int}}(undef, k)
    for i = 1:k
        gs[i] = Int[]
    end

    warned = false
    n = length(xs)
    for i = 1:n
        @inbounds x = xs[i]
        if x < 1 || x > k
            if warning && !warned
                warn("Some labels are out of bound.")
            end
        end
        push!(gs[x], i)
    end
    return gs
end


function groupindices(lmap::LabelMap{T}, xs::AbstractArray{T}) where T
    k = length(lmap)
    gs = Array{Vector{Int}}(undef, k)
    for i = 1:k
        gs[i] = Int[]
    end
    n = length(xs)
    for i = 1:n
        @inbounds l = labelencode(lmap, xs[i])
        push!(gs[l], i)
    end
    return gs
end
