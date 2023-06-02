# Performance evaluation

## correctrate & errorrate

correctrate(gt::IntegerVector, r::IntegerVector) = counteq(gt, r) / length(gt)
errorrate(gt::IntegerVector, r::IntegerVector) = countne(gt, r) / length(gt)

## confusion matrix
function confusmat(gts::IntegerVector, preds::IntegerVector)
    n = length(gts)
    length(preds) == n || throw(DimensionMismatch("Inconsistent lengths."))

    gtslbl = sort(unique(gts))
    k = length(gtslbl)

    lookup = Dict(reverse.(enumerate(gtslbl)|> collect))
    R = zeros(Int, k, k)
    for i = 1:n
        @inbounds g = lookup[gts[i]]
        @inbounds p = lookup[preds[i]]
        R[g, p] += 1
    end
    return R
end

## counthits & hitrate

function counthits(gt::IntegerVector, rklst::IntegerMatrix, k::Integer)
    n = length(gt)
    size(rklst, 2) == n || throw(DimensionMismatch("Input dimensions mismatch."))
    m = min(size(rklst, 1), Int(k))

    cnt = 0
    @inbounds for j = 1:n
        rj = view(rklst, :, j)
        gj = gt[j]
        for i = 1:m
            if rj[i] == gj
                cnt += 1
                break
            end
        end
    end
    return cnt::Int
end

function counthits(gt::IntegerVector, rklst::IntegerMatrix, ks::IntegerVector)
    n = length(gt)
    size(rklst, 2) == n || throw(DimensionMismatch("Input dimensions mismatch."))
    issorted(ks) || throw(DimensionMismatch("ks must be sorted."))

    m = min(size(rklst, 1), ks[end])
    nk = length(ks)
    cnts = zeros(Int, nk)
    @inbounds for j = 1:n
        rj = view(rklst, :, j)
        gj = gt[j]
        for i = 1:m
            if rj[i] == gj
                ik = 1
                while ks[ik] < i; ik += 1; end
                while ik <= nk
                    cnts[ik] += 1
                    ik += 1
                end
                break
            end
        end
    end
    return cnts
end


hitrate(gt::IntegerVector, rklst::IntegerMatrix, k::Integer) =
    (counthits(gt, rklst, k) / length(gt))::Float64

function hitrates(gt::IntegerVector, rklst::IntegerMatrix, ks::IntegerVector)
    n = length(gt)
    h = counthits(gt, rklst, ks)
    nk = length(ks)
    r = Array{Float64}(undef, nk)
    for i = 1:nk
        r[i] = h[i] / n
    end
    return r
end


## ROC

struct ROCNums{T<:Real}
    p::T    # positive in ground-truth
    n::T    # negative in ground-truth
    tp::T   # correct positive prediction
    tn::T   # correct negative prediction
    fp::T   # (incorrect) positive prediction when ground-truth is negative
    fn::T   # (incorrect) negative prediction when ground-truth is positive
end

function show(io::IO, x::ROCNums)
    println(io, "$(typeof(x))")
    println(io, "  p = $(x.p)")
    println(io, "  n = $(x.n)")
    println(io, "  tp = $(x.tp)")
    println(io, "  tn = $(x.tn)")
    println(io, "  fp = $(x.fp)")
    println(io, "  fn = $(x.fn)")
end

true_positive(x::ROCNums) = x.tp
true_negative(x::ROCNums) = x.tn
false_positive(x::ROCNums) = x.fp
false_negative(x::ROCNums) = x.fn

true_positive_rate(x::ROCNums) = x.tp / x.p
true_negative_rate(x::ROCNums) = x.tn / x.n
false_positive_rate(x::ROCNums) = x.fp / x.n
false_negative_rate(x::ROCNums) = x.fn / x.p

recall(x::ROCNums) = true_positive_rate(x)
precision(x::ROCNums) = x.tp / (x.tp + x.fp)

f1score(x::ROCNums) = (tp2 = x.tp + x.tp; tp2 / (tp2 + x.fp + x.fn) )


## roc: produce a single instance of ROCNums from predictions

_ispos(x::Bool) = x
_ispos(x::Real) = x > zero(x)

function _roc(gt::IntegerVector, pr)
    len = length(gt)
    length(pr) == len || throw(DimensionMismatch("Inconsistent lengths."))

    p = 0
    n = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i = 1:len
        @inbounds gi = gt[i]
        @inbounds ri = pr[i]
        if _ispos(gi)   # gt = true
            p += 1
            if ri == gi
                tp += 1
            elseif ri == 0
                fn += 1
            end
        else        # gt = false
            n += 1
            if _ispos(ri)
                fp += 1
            else
                tn += 1
            end
        end
    end

    return ROCNums{Int}(p, n, tp, tn, fp, fn)
end

# compute roc numbers based on prediction
roc(gt::IntegerVector, pr::IntegerVector) = _roc(gt, pr)

##
#   BinaryThresPredVec immutates a vector:
#
#   v[i] := scores[i] < thres ? 0 : 1
#
struct BinaryThresPredVec{ScoreVec <: RealVector,
                          T <: Real,
                          Ord <: Ordering}
    scores::ScoreVec
    thres::T
    ord::Ord
end

length(v::BinaryThresPredVec) = length(v.scores)
getindex(v::BinaryThresPredVec, i::Integer) = !lt(v.ord, v.scores[i], v.thres)

# compute roc numbers based on scores & threshold
roc(gt::IntegerVector, scores::RealVector, t::Real, ord::Ordering) =
    _roc(gt, BinaryThresPredVec(scores, t, ord))

roc(gt::IntegerVector, scores::RealVector, thres::Real) =
    roc(gt, scores, thres, Forward)

##
#   ThresPredVec immutates a vector:
#
#   v[i] := scores[i] < thres ? 0 : preds[i]
#
struct ThresPredVec{PredVec <: IntegerVector,
                    ScoreVec <: RealVector,
                    T <: Real,
                    Ord <: Ordering}

    preds::PredVec
    scores::ScoreVec
    thres::T
    ord::Ordering
end

function ThresPredVec(
    preds::PVec, scores::SVec, thres::T, ord::Ord) where {PVec<:IntegerVector,SVec<:RealVector,T<:Real,Ord<:Ordering}
    n = length(preds)
    length(scores) == n || throw(DimensionMismatch("Inconsistent lengths."))
    ThresPredVec{PVec,SVec,T,Ord}(preds, scores, thres, ord)
end

length(v::ThresPredVec) = length(v.preds)
getindex(v::ThresPredVec, i::Integer) = ifelse(lt(v.ord, v.scores[i], v.thres), 0, v.preds[i])

# compute roc numbers based on predictions & scores & threshold
roc(gt::IntegerVector, preds::Tuple{PV,SV}, t::Real, ord::Ordering) where {PV<:IntegerVector,SV<:RealVector} =
    _roc(gt, ThresPredVec(preds..., t, ord))

roc(gt::IntegerVector, preds::Tuple{PV,SV}, thres::Real) where {PV<:IntegerVector,SV<:RealVector} =
    roc(gt, preds, thres, Forward)


## roc: produces a series of ROCNums instances with multiple thresholds

# find_thresbin
#
#  x < threshold[1] --> 1
#  threshold[i] <= x < threshold[i+1] --> i+1
#  x >= threshold[n] --> n+1
#
function find_thresbin(x::Real, thresholds::RealVector, ord::Ordering)
    n = length(thresholds)
    r = 1
    if !lt(ord, x, thresholds[1])
        l = 1
        r = n + 1
        while l + 1 < r
            m = (l + r) >> 1
            if lt(ord, x, thresholds[m])
                r = m
            else
                l = m
            end
        end
    end
    return r::Int
end

find_thresbin(x::Real, thresholds::RealVector) = find_thresbin(x, thresholds, Forward)

lin_thresholds(scores::RealArray, n::Integer, ord::ForwardOrdering) =
    ((s0, s1) = extrema(scores); intv = (s1 - s0) / (n-1); s0:intv:s1)

lin_thresholds(scores::RealArray, n::Integer, ord::ReverseOrdering{ForwardOrdering}) =
    ((s0, s1) = extrema(scores); intv = (s0 - s1) / (n-1); s1:intv:s0)

# roc for binary predictions
function roc(gt::IntegerVector, scores::RealVector, thresholds::RealVector, ord::Ordering)
    issorted(thresholds, ord) || error("thresholds must be sorted w.r.t. the given ordering.")

    ns = length(scores)
    nt = length(thresholds)

    # scan scores and classify them into bins
    hp = zeros(Int, nt + 1)
    hn = zeros(Int, nt + 1)
    p = 0
    n = 0
    for i = 1:ns
        @inbounds s = scores[i]
        @inbounds g = gt[i]
        k = find_thresbin(s, thresholds, ord)
        if _ispos(g)
            hp[k] += 1
            p += 1
        else
            hn[k] += 1
            n += 1
        end
    end

    # produce results
    r = Array{ROCNums{Int}}(undef, nt)
    fn = 0
    tn = 0
    @inbounds for i = 1:nt
        fn += hp[i]
        tn += hn[i]
        tp = p - fn
        fp = n - tn
        r[i] = ROCNums{Int}(p, n, tp, tn, fp, fn)
    end
    return r
end

roc(gt::IntegerVector, scores::RealVector, thresholds::RealVector) = roc(gt, scores, thresholds, Forward)

roc(gt::IntegerVector, scores::RealVector, n::Integer, ord::Ordering) =
    roc(gt, scores, lin_thresholds(scores, n, ord), ord)

roc(gt::IntegerVector, scores::RealVector, n::Integer) = roc(gt, scores, n, Forward)

roc(gt::IntegerVector, scores::RealVector, ord::Ordering) = roc(gt, scores, 100, ord)
roc(gt::IntegerVector, scores::RealVector) = roc(gt, scores, Forward)

# roc for multi-way predictions
function roc(
    gt::IntegerVector, preds::Tuple{PV,SV}, thresholds::RealVector, ord::Ordering) where {PV<:IntegerVector,SV<:RealVector}

    issorted(thresholds, ord) || error("thresholds must be sorted w.r.t. the given ordering.")
    pr::PV = preds[1]
    scores::SV = preds[2]

    ns = length(scores)
    nt = length(thresholds)

    # scan scores and classify them into bins
    hp = zeros(Int, nt + 1)
    hn = zeros(Int, nt + 1)
    htp = zeros(Int, nt + 1)

    p = 0
    n = 0
    tp = 0

    for i = 1:ns
        @inbounds pi = pr[i]
        @inbounds si = scores[i]
        @inbounds gi = gt[i]
        k = find_thresbin(si, thresholds, ord)
        if _ispos(gi)
            hp[k] += 1
            p += 1

            if pi == gi
                htp[k] += 1
                tp += 1
            end
        else
            hn[k] += 1
            n += 1
        end
    end

    # produce results
    r = Array{ROCNums{Int}}(undef, nt)
    fn = 0
    tn = 0
    @inbounds for i = 1:nt
        fn += hp[i]
        tn += hn[i]
        tp -= htp[i]
        fp = n - tn
        r[i] = ROCNums{Int}(p, n, tp, tn, fp, fn)
    end
    return r
end

roc(gt::IntegerVector, preds::Tuple{PV,SV}, thresholds::RealVector) where {PV<:IntegerVector, SV<:RealVector} =
    roc(gt, preds, thresholds, Forward)

roc(gt::IntegerVector, preds::Tuple{PV,SV}, n::Integer, ord::Ordering) where {PV<:IntegerVector, SV<:RealVector} =
    roc(gt, preds, lin_thresholds(preds[2],n,ord), ord)

roc(gt::IntegerVector, preds::Tuple{PV,SV}, n::Integer) where {PV<:IntegerVector, SV<:RealVector} =
    roc(gt, preds, n, Forward)

roc(gt::IntegerVector, preds::Tuple{PV,SV}, ord::Ordering) where {PV<:IntegerVector, SV<:RealVector} =
    roc(gt, preds, 100, ord)

roc(gt::IntegerVector, preds::Tuple{PV,SV}) where {PV<:IntegerVector, SV<:RealVector} =
    roc(gt, preds, Forward)
