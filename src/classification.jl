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

classify(x::RealMatrix, ord::Ordering) = classify!(Array(Int, size(x,2)), x, ord)
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

function classify_withscores{T<:Real}(x::RealMatrix{T}, ord::Ordering)
    n = size(x, 2)
    r = Array(Int, n)
    s = Array(T, n)
    return classify_withscores!(r, s, x, ord)
end

classify_withscores{T<:Real}(x::RealMatrix{T}) = classify_withscores(x, Forward)


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

classify(x::RealMatrix, t::Real, ord::Ordering) = classify!(Array(Int, size(x,2)), x, t, ord)
classify(x::RealMatrix, t::Real) = classify(x, t, Forward)


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
        println(io, "[$i] $v")
    end
end


# build a label map (value -> label) from a sequences of values
function labelmap{T}(xs::AbstractArray{T})
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


## class encodings

abstract ClassEncoding
abstract BinaryClassEncoding <: ClassEncoding
abstract MultinomialClassEncoding <: ClassEncoding

## binary class encodings

immutable ZeroOneClassEncoding{T} <: BinaryClassEncoding
  labelmap::LabelMap{T}

  function ZeroOneClassEncoding(labelmap::LabelMap{T})
    numLabels = length(labelmap.vs)
    if numLabels != 2
      throw(ArgumentError("The given target vector must have exactly two classes"))
    end
    new(labelmap)
  end
end

ZeroOneClassEncoding{T}(targets::Vector{T}) =
  ZeroOneClassEncoding{T}(labelmap(targets))

immutable SignedClassEncoding{T} <: BinaryClassEncoding
  labelmap::LabelMap{T}

  function SignedClassEncoding(labelmap::LabelMap{T})
    numLabels = length(labelmap.vs)
    if numLabels != 2
      throw(ArgumentError("The given target vector must have exactly two classes"))
    end
    new(labelmap)
  end
end

SignedClassEncoding{T}(targets::Vector{T}) =
  SignedClassEncoding{T}(labelmap(targets))

## multinomial class encodings

immutable MultivalueClassEncoding{T} <: MultinomialClassEncoding
  labelmap::LabelMap{T}
  nlabels::Int
  zeroBased::Bool

  function MultivalueClassEncoding(labelmap::LabelMap{T}, zeroBased = false)
    numLabels = length(labelmap.vs)
    if numLabels < 2
      throw(ArgumentError("The given target vector has less than two classes"))
    end
    new(labelmap, numLabels, zeroBased)
  end
end

MultivalueClassEncoding{T}(targets::Vector{T}; zero_based = false) =
  MultivalueClassEncoding{T}(labelmap(targets), zero_based)

immutable OneOfKClassEncoding{T} <: MultinomialClassEncoding
  labelmap::LabelMap{T}
  nlabels::Int

  function OneOfKClassEncoding(labelmap::LabelMap{T})
    numLabels = length(labelmap.vs)
    if numLabels < 2
      throw(ArgumentError("The given target vector has less than two classes"))
    end
    new(labelmap, numLabels)
  end
end

OneOfKClassEncoding{T}(targets::Vector{T}) =
  OneOfKClassEncoding{T}(labelmap(targets))

typealias OneHotClassEncoding OneOfKClassEncoding

## display class encodings

function getLabelString{T}(labelmap::LabelMap{T})
  labels = labelmap.vs
  c = length(labels)
  if c > 10
    labels = labels[1:10]
    labelString = string(join(labels, ", "), ", ... [TRUNC]")
  else
    labelString = join(labels, ", ")
  end
end

function show{T}(io::IO, classEncoding::ZeroOneClassEncoding{T})
  labelString = getLabelString(classEncoding.labelmap)
  print(io,
        """
        ZeroOneClassEncoding (Binary) to {0, 1}
          .labelmap  ...  encoding for: {$labelString}""")
end

function show{T}(io::IO, classEncoding::SignedClassEncoding{T})
  labelString = getLabelString(classEncoding.labelmap)
  print(io,
        """
        SignedClassEncoding (Binary) to {-1, 1}
          .labelmap  ...  encoding for: {$labelString}""")
end

function show{T}(io::IO, classEncoding::MultivalueClassEncoding{T})
  c = classEncoding.nlabels
  zB = classEncoding.zeroBased
  labelString = getLabelString(classEncoding.labelmap)
  print(io,
        """
        MultivalueClassEncoding (Multinomial) to range $((1:c) - zB*1)
          .nlabels   ...  $c classes
          .labelmap  ...  encoding for: {$labelString}""")
end

function show{T}(io::IO, classEncoding::OneOfKClassEncoding{T})
  c = classEncoding.nlabels
  labelString = getLabelString(classEncoding.labelmap)
  print(io,
        """
        OneOfKClassEncoding (Multinomial) to one-out-of-$c hot-vector
          .nlabels   ...  $c classes
          .labelmap  ...  encoding for: {$labelString}""")
end


# use a map to encode discrete values into labels
labelencode{T}(lmap::LabelMap{T}, x) = lmap.v2i[convert(T, x)]
labelencode{T}(lmap::LabelMap{T}, xs::AbstractArray{T}) =
    reshape(Int[labelencode(lmap, x) for x in xs], size(xs))

# decode the label to the associated discrete value
labeldecode{T}(lmap::LabelMap{T}, y::Int) = lmap.vs[y]
labeldecode{T}(lmap::LabelMap{T}, ys::AbstractArray{Int}) =
    reshape(T[labeldecode(lmap, y) for y in ys], size(ys))

# encoding and decoding the labels for class encodings
function labelencode{T}(classEncoding::ZeroOneClassEncoding{T}, targets::Vector{T})
  indicies = labelencode(classEncoding.labelmap, targets)
  indicies - 1
end

function labeldecode{T}(classEncoding::ZeroOneClassEncoding{T}, values::Vector{Int})
  indicies = values + 1
  labeldecode(classEncoding.labelmap, indicies)
end

function labelencode{T}(classEncoding::SignedClassEncoding{T}, targets::Vector{T})
  indicies = labelencode(classEncoding.labelmap, targets)
  round(Integer,2(indicies - 1.5))
end

function labeldecode{T}(classEncoding::SignedClassEncoding{T}, values::Vector{Int})
  indicies = round(Integer,(values / 2.) + 1.5)
  labeldecode(classEncoding.labelmap, indicies)
end

function labelencode{T}(classEncoding::MultivalueClassEncoding{T}, targets::Vector{T})
  labelencode(classEncoding.labelmap, targets) - classEncoding.zeroBased*1
end

function labeldecode{T}(classEncoding::MultivalueClassEncoding{T}, values::Vector{Int})
  labeldecode(classEncoding.labelmap, values + classEncoding.zeroBased*1)
end

function labelencode{T}(classEncoding::OneOfKClassEncoding{T}, targets::Vector{T})
  indicies = labelencode(classEncoding.labelmap, targets)
  convert(Matrix{Int}, indicatormat(indicies))'
end

function labeldecode{T}(classEncoding::OneOfKClassEncoding{T}, values::Matrix{Int})
  numLabels = classEncoding.nlabels
  indicies = convert(Vector{Int}, values * collect(1:numLabels))
  labeldecode(classEncoding.labelmap, indicies)
end


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

function groupindices{T}(classEncoding::ClassEncoding, targets::Vector{T})
  groupindices(classEncoding.labelmap, targets)
end


# class distribution

function classDistribution{T}(classEncoding::ClassEncoding, targets::Vector{T})
  classEncoding.labelmap.vs, map(length,groupindices(classEncoding, targets))
end
