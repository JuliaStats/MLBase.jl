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

# compute roc numbers based on prediction
function rocnums(gt::IntegerVector, pr::IntegerVector)
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
        if gi > 0   # gt = true
            p += 1
            if ri == gi
                tp += 1
            elseif ri == 0
                fn += 1
            end
        else        # gt = false
            n += 1
            if ri <= 0
                tn += 1
            else
                fp += 1
            end
        end
    end

    return ROCNums{Int}(p, n, tp, tn, fp, fn)
end

# compute roc numbers based on scores & threshold
function rocnums(gt::IntegerVector, pr::IntegerVector, scores::RealVector, 
        thres::Real, op::ToMaxOrMin)

    len = length(gt)
    length(pr) == length(scores) == len || 
        throw(DimensionMismatch("Inconsistent dimensions."))

    p = 0
    n = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i = 1:len
        @inbounds gi = gt[i]        
        @inbounds ri = ifelse(better(scores[i], thres, op), pr[i], 0)
        if gi > 0   # gt = true
            p += 1
            if ri == gi
                tp += 1
            elseif ri == 0
                fn += 1
            end
        else        # gt = false
            n += 1
            if ri <= 0
                tn += 1
            else
                fp += 1
            end
        end
    end

    return ROCNums{Int}(p, n, tp, tn, fp, fn)
end

rocnums(gt::IntegerVector, pr::IntegerVector, scores::RealVector, thres::Real) =
    rocnums(gt, pr, scores, thres, to_max())

