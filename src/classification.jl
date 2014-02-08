# manipulation of class labels

## simple classification

classify(x::RealVector; to_max::Bool=true) = to_max ? indmax(x) : indmin(x)

function classify!(r::IntegerVector, x::RealMatrix; to_max::Bool=true)
    m = size(x, 1)
    n = size(x, 2)
    length(r) == n || throw(DimensionMismatch("Mismatched length of r."))
    if to_max
        for j = 1:n
            @inbounds r[j] = indmax(view(x, :, j))
        end
    else
        for j = 1:n
            @inbounds r[j] = indmin(view(x, :, j))
        end
    end
    return r
end

classify(x::RealMatrix; to_max::Bool=true) = classify!(Array(Int, size(x,2)), x; to_max=to_max)

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

