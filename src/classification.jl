# manipulation of class labels

## simple classification

type ToMax end
type ToMin end
typealias ToMaxOrMin Union(ToMax,ToMin)

to_max() = ToMax()
to_min() = ToMin()

# classify

classify(x::RealVector, ::ToMax) = indmax(x)
classify(x::RealVector, ::ToMin) = indmin(x)
classify(x::RealVector) = classify(x, to_max())

function classify!(r::IntegerVector, x::RealMatrix, op::ToMaxOrMin)
    m = size(x, 1)
    n = size(x, 2)
    length(r) == n || throw(DimensionMismatch("Mismatched length of r."))
    for j = 1:n
        @inbounds r[j] = classify(view(x,:,j), op)
    end
    return r
end

classify!(r::IntegerVector, x::RealMatrix) = classify!(r, x, to_max())

classify(x::RealMatrix, op::ToMaxOrMin) = classify!(Array(Int, size(x,2)), x, op)
classify(x::RealMatrix) = classify(x, to_max())

# classify with score(s)

classify_withscore(x::RealVector, op::ToMaxOrMin) = 
    (i = classify(x, op); (i, x[i]))

classify_withscore(x::RealVector) = classify_withscore(x, to_max())

function classify_withscores!(r::IntegerVector, s::RealVector, x::RealMatrix, op::ToMaxOrMin)
    m = size(x, 1)
    n = size(x, 2)
    length(r) == n || throw(DimensionMismatch("Mismatched length of r."))
    for j = 1:n
        xj = view(x, :, j)
        k = classify(xj, op)
        @inbounds r[j] = k
        @inbounds s[j] = xj[k]
    end
    return (r, s)
end

classify_withscores!(r::IntegerVector, s::RealVector, x::RealMatrix) = 
    classify_withscores!(r, s, x, to_max())

function classify_withscores{T<:Real}(x::RealMatrix{T}, op::ToMaxOrMin)
    n = size(x, 2)
    r = Array(Int, n)
    s = Array(T, n)
    return classify_withscores!(r, s, x, op)
end

classify_withscores{T<:Real}(x::RealMatrix{T}) = classify_withscores(x, to_max())


# classify with threshold

classify(x::RealVector, t::Real, ::ToMax) = (i = indmax(x); ifelse(x[i] >= t, i, 0))
classify(x::RealVector, t::Real, ::ToMin) = (i = indmin(x); ifelse(x[i] <= t, i, 0))
classify(x::RealVector, t::Real) = classify(x, t, to_max())

function classify!(r::IntegerVector, x::RealMatrix, t::Real, op::ToMaxOrMin)
    m = size(x, 1)
    n = size(x, 2)
    length(r) == n || throw(DimensionMismatch("Mismatched length of r."))
    for j = 1:n
        @inbounds r[j] = classify(view(x,:,j), t, op)
    end
    return r
end

classify!(r::IntegerVector, x::RealMatrix, t::Real) = classify!(r, x, t, to_max())
classify(x::RealMatrix, t::Real, op::ToMaxOrMin) = classify!(Array(Int, size(x,2)), x, t, op)
classify(x::RealMatrix, t::Real) = classify(x, t, to_max())  


## label map

immutable LabelMap{K}
    vs::Array{K}
    v2i::Dict{K,Int}
end

function LabelMap{K}(vs::Array{K}, v2i::Dict{K,Int})
    length(vs) == length(v2i) || throw(DimensionMismatch("Lengths of vs and v2i mismatch"))
    LabelMap{K}(vs, v2i)
end

length(lmap::LabelMap) = length(lmap.vs)
keys(lmap::LabelMap) = lmap.vs

function show(io::IO, lmap::LabelMap)
    n = length(lmap)
    println(io, "LabelMap (with $n labels):")
    for (i, v) in enumerate(keys(lmap))
        println(io, "[$i]: $v")
    end
end


# build a label map (value -> label) from a sequences of values
function labelmap{T}(xs::AbstractArray{T})
    l = 0
    vs = T[]
    v2i = (T=>Int)[]
    for x in xs
        if !haskey(v2i, x)
            push!(vs, x)
            v2i[x] = (l += 1)
        end
    end
    return LabelMap(vs, v2i)
end

# use a map to encode discrete values into labels
labelencode{T}(lmap::LabelMap{T}, x) = lmap.v2i[convert(T, x)]
labelencode{T}(lmap::LabelMap{T}, xs::AbstractArray{T}) = 
    reshape(Int[labelencode(lmap, x) for x in xs], size(xs))

## group labels

function groupindices(k::Int, xs::IntegerVector; warning::Bool=true)
    gs = Array(Vector{Int}, k)
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


function groupindices{T}(lmap::LabelMap{T}, xs::AbstractArray{T})
    k = length(lmap)
    gs = Array(Vector{Int}, k)
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

