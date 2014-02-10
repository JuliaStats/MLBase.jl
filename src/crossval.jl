# Cross validation

## cross validation generators

abstract CrossValGenerator

# K-fold 

immutable Kfold <: CrossValGenerator
    permseq::Vector{Int}
    k::Int
    coeff::Float64

    function Kfold(n::Int, k::Int)
        2 <= k <= n || error("The value of k must be in [2, length(a)].")
        new(randperm(n), k, n / k)
    end
end

length(c::Kfold) = c.k

immutable KfoldState
    i::Int      # the i-th of the subset 
    s::Int      # starting index
    e::Int      # ending index
end

start(c::Kfold) = KfoldState(1, 1, iround(c.coeff))
next(c::Kfold, s::KfoldState) = 
    (i = s.i+1; (sort!(c.permseq[s.s:s.e]), KfoldState(i, s.e+1, iround(c.coeff * i))))
done(c::Kfold, s::KfoldState) = (s.i > c.k)


# LOOCV (Leave-one-out cross-validation)

function leave_one_out(n::Int, i::Int)
    @assert 1 <= i <= n
    x = Array(Int, n-1)
    for j = 1:i-1
        x[j] = j
    end
    for j = i+1:n
        x[j-1] = j
    end
    return x
end

immutable LOOCV <: CrossValGenerator
    n::Int
end

length(c::LOOCV) = c.n

start(c::LOOCV) = 1
next(c::LOOCV, s::Int) = (leave_one_out(c.n, s), s+1)
done(c::LOOCV, s::Int) = (s > c.n)


# Repeated random sub-sampling

immutable RandomSub <: CrossValGenerator
    n::Int    # total length
    sn::Int   # length of each subset
    k::Int    # number of subsets
end

length(c::RandomSub) = c.k

start(c::RandomSub) = 1
next(c::RandomSub, s::Int) = (sample(1:c.n, c.sn; replace=false), s+1)
done(c::RandomSub, s::Int) = (s > c.k)


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
#  n:   the number of samples in the whole data set
#
#  gen: an iterable object (e.g. an instance of CrossValGenerator)
#       where each element is a vector of indices
#
#  ord: ordering of the score
#
#  This function returns a tuple (best_model, best_score, best_traininds)
#
function cross_validate(estfun::Function, evalfun::Function, n::Integer, gen, ord::Ordering)
    best_model = nothing
    best_score = NaN   
    best_inds = Int[]
    first = true

    for train_inds in gen
        test_inds = setdiff(1:n, train_inds)
        model = estfun(train_inds)
        score = evalfun(model, test_inds)
        if first || lt(ord, best_score, score)
            best_model = model
            best_score = score            
            best_inds = train_inds
            first = false
        end
    end

    return (best_model, best_score, best_inds)
end

cross_validate(estfun::Function, evalfun::Function, n::Integer, gen) = 
    cross_validate(estfun, evalfun, n, gen, Forward)



