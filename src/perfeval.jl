# Performance evaluation

## correctrate & errorrate

correctrate(gt::IntegerVector, r::IntegerVector) = counteq(gt, r) / length(gt)
errorrate(gt::IntegerVector, r::IntegerVector) = countne(gt, r) / length(gt)

## ROC

immutable ROCNums{T<:Real}
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

## rocnums

_ispos(x::Bool) = x
_ispos(x::Real) = x > zero(x)

function _rocnums(gt::IntegerVector, pr)
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
rocnums(gt::IntegerVector, pr::IntegerVector) = _rocnums(gt, pr)

# compute roc numbers based on scores & threshold
immutable BinaryThresPredVec{ScoreVec<:RealVector,T,Op<:ToMaxOrMin}
    scores::ScoreVec
    thres::T
    op::Op
end

BinaryThresPredVec{SVec<:RealVector,T<:Real,Op<:ToMaxOrMin}(scores::SVec, thres::T, op::Op) = 
    BinaryThresPredVec{SVec,T,Op}(scores, thres, op)

length(v::BinaryThresPredVec) = length(v.scores)
getindex(v::BinaryThresPredVec, i::Integer) = better(v.scores[i], v.thres, v.op)

rocnums(gt::IntegerVector, scores::RealVector, t::Real, op::ToMaxOrMin) = 
    _rocnums(gt, BinaryThresPredVec(scores, t, op))

rocnums(gt::IntegerVector, scores::RealVector, thres::Real) =
    rocnums(gt, scores, thres, to_max())


# compute roc numbers based on predictions & scores & threshold
immutable ThresPredVec{PredVec<:IntegerVector,ScoreVec<:RealVector,T,Op<:ToMaxOrMin}
    preds::PredVec
    scores::ScoreVec
    thres::T
    op::Op
end

function ThresPredVec{PVec<:IntegerVector,SVec<:RealVector,T<:Real,Op<:ToMaxOrMin}(
    preds::PVec, scores::SVec, thres::T, op::Op)
    n = length(preds)
    length(scores) == n || throw(DimensionMismatch("Inconsistent lengths."))
    ThresPredVec{PVec,SVec,T,Op}(preds, scores, thres, op)
end

length(v::ThresPredVec) = length(v.preds)
getindex(v::ThresPredVec, i::Integer) = ifelse(better(v.scores[i], v.thres, v.op), v.preds[i], 0)::Int

rocnums(gt::IntegerVector, pr::IntegerVector, scores::RealVector, t::Real, op::ToMaxOrMin) = 
    _rocnums(gt, ThresPredVec(pr, scores, t, op))

rocnums(gt::IntegerVector, pr::IntegerVector, scores::RealVector, thres::Real) =
    rocnums(gt, pr, scores, thres, to_max())

