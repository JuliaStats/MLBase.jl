# Cross validation

## cross validation generators

abstract type CrossValGenerator end

# K-fold

struct Kfold <: CrossValGenerator
    permseq::Vector{Int}
    k::Int
    coeff::Float64

    function Kfold(n::Int, k::Int)
        2 <= k <= n || error("The value of k must be in [2, length(a)].")
        new(randperm(n), k, n / k)
    end
end

length(c::Kfold) = c.k

struct KfoldState
    i::Int      # the i-th of the subset
    s::Int      # starting index
    e::Int      # ending index
end

# A version check allows to maintain compatibility with earlier versions
function Base.iterate(c::Kfold, state::KfoldState=KfoldState(1, 1, round.(Integer, c.coeff)))
    i, s, e = state.i, state.s, state.e
    (i > c.k) && return nothing
    i += 1
    sd = setdiff(1:length(c.permseq), c.permseq[s:e])
    kst = KfoldState(i, e + 1, round.(Integer, c.coeff * i))
    return (i; (sd, kst))
end

# Stratified K-fold

struct StratifiedKfold <: CrossValGenerator
    n::Int                         #Total number of observations
    permseqs::Vector{Vector{Int}}  #Vectors of vectors of indexes for each stratum
    k::Int                         #Number of splits
    coeffs::Vector{Float64}        #About how many observations per strata are in a val set
    function StratifiedKfold(strata, k)
        2 <= k <= length(strata) || error("The value of k must be in [2, length(strata)].")
        strata_labels, permseqs = unique_inverse(strata)
        map(shuffle!, permseqs)
        coeffs = Float64[]
        for (stratum, permseq) in zip(strata_labels, permseqs)
            k <= length(permseq) || error("k is greater than the length of stratum $stratum")
            push!(coeffs, length(permseq) / k)
        end
        new(length(strata), permseqs, k, coeffs)
    end
end

length(c::StratifiedKfold) = c.k

function Base.iterate(c::StratifiedKfold, s::Int=1)
    (s > c.k) && return nothing
    r = Int[]
    for (permseq, coeff) in zip(c.permseqs, c.coeffs)
        a, b = round.(Integer, [s-1, s] .* coeff)
        append!(r, view(permseq, a+1:b))
    end
    return setdiff(1:c.n, r), s+1
end

# LOOCV (Leave-one-out cross-validation)

function leave_one_out(n::Int, i::Int)
    @assert 1 <= i <= n
    x = Array{Int}(undef, n - 1)
    for j = 1:i-1
        x[j] = j
    end
    for j = i+1:n
        x[j-1] = j
    end
    return x
end

struct LOOCV <: CrossValGenerator
    n::Int
end

length(c::LOOCV) = c.n

function iterate(c::LOOCV, s::Int=1)
    (s > c.n) && return nothing
    return (leave_one_out(c.n, s), s + 1)
end

# Repeated random sub-sampling

struct RandomSub <: CrossValGenerator
    n::Int    # total length
    sn::Int   # length of each subset
    k::Int    # number of subsets
end

length(c::RandomSub) = c.k

function iterate(c::RandomSub, s::Int=1)
    (s > c.k) && return nothing
    return (sort!(sample(1:c.n, c.sn; replace=false)), s+1)
end

# Stratified repeated random sub-sampling

struct StratifiedRandomSub <: CrossValGenerator
    idxs::Vector{Vector{Int}}    # total length
    sn::Int                      # length of subset
    sns::Vector{Int}             # num from each stratum for each subset
    k::Int                       # number of subsets
    function StratifiedRandomSub(strata, sn, k)
        n = length(strata)
        strata_labels, idxs = unique_inverse(strata)
        sn >= length(strata_labels) || error("sn is too small for all strata to be represented")
        lengths_ord = sortperm(map(length, idxs))
        sns = zeros(Int, n)
        remaining_n = n   # total in the strata we haven't seen yet
        remaining_sn = sn # total room in the subset that hasn't been "assinged" to a stratum yet
        #loop through strata from smallest to largest, making sure there is at least one idx
        #from each strata in each subset:
        for stratum_num in lengths_ord
            stratum_n = length(idxs[stratum_num])
            remaining_proportion = remaining_sn/remaining_n
            stratum_sn = max(round.(Integer, remaining_proportion*stratum_n), 1)
            remaining_n -= stratum_n
            remaining_sn -= stratum_sn
            sns[stratum_num] = stratum_sn
        end
        #@assert mapreduce(sum, +, sns) == sn
        new(idxs, sn, sns, k)
    end
end

length(c::StratifiedRandomSub) = c.k

function iterate(c::StratifiedRandomSub, s::Int=1)
    (s > c.k) && return nothing
    idxs = Array{Int}(undef, 0)
    sizehint!(idxs, c.sn)
    for (stratum_sn, stratum_idxs) in zip(c.sns, c.idxs)
        append!(idxs, sample(stratum_idxs, stratum_sn, replace=false))
    end
    return (sort!(idxs), s + 1)
end

## Cross validation algorithm
#
#  estfun: model estimation function
#
#          model = estfun(train_inds)
#
#          it takes as input the indices of
#          the samples for training, and returns
#          a trained model.
#
#  evalfun: model evaluation function
#
#           v = evalfun(model, test_inds)
#
#           it applies a trained model to a subset
#           of samples whose indices are given in
#           test_inds, and returns an overall score.
#
#  n:   the total number of samples
#
#  gen: an iterable object (e.g. an instance of CrossValGenerator)
#       where each element is a vector of indices
#
#  This function returns a tuple a vector of validation scores
#
function cross_validate(estfun::Function, evalfun::Function, n::Int, gen)
    best_model = nothing
    best_score = NaN
    best_inds = Int[]
    first = true

    scores = Float64[]
    for (i, train_inds) in enumerate(gen)
        test_inds = setdiff(1:n, train_inds)
        model = estfun(train_inds)
        score = evalfun(model, test_inds)
        push!(scores, score)
    end
    return scores
end

cross_validate(estfun::Function, evalfun::Function, n::Integer, gen) =
    cross_validate(estfun, evalfun, n, gen, Forward)
