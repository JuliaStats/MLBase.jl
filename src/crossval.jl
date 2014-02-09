# Cross validation


## cross validation strategies

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



