module MLBase

    using ArrayViews
    using StatsBase

    import Base: length, show, keys, precision, length, getindex
    import Base: start, next, done
    import Base.Order: lt, Ordering, ForwardOrdering, ReverseOrdering, Forward, Reverse
    import StatsBase: RealVector, IntegerVector, RealMatrix, IntegerMatrix, RealArray
    import StatsBase: sample

    export

    # reexport from Base.Order
    Forward, Reverse,

    # reexport some functions from StatsBase
    counts, addcounts!, countmap, proportions, 
    indicatormat, sample, weights, WeightVec, wmean,

    # utils
    repeach,        # repeat each element in a vector 
    repeachcol,     # repeat each column in a matrix
    repeachrow,     # repeat each row in a matrix
    counteq,        # count the number of equal pairs
    countne,        # count the number of non-equal pairs

    # datapre
    Standardize,    # the type to represent a standardizing transform

    indim,          # input dimension of a transform
    outdim,         # output dimension of a transform 
    estimate,       # estimate a model or transformation
    transform,      # apply a transformation to data
    transform!,     # apply a transformation to data in place
    standardize,    # estimate and apply a standardization
    standardize!,   # estimate and apply a standardization in place
        
    # classification
    LabelMap,       # a type to represent a label map

    classify,       # predict class label(s) based on score values
    classify!,      # inplace version of classify
    classify_withscore,     # classify with additional output of best score
    classify_withscores,    # classify with additional output of best scores
    classify_withscores!,   # inplace version of classify_withscores
    labelmap,       # construct a label map from a list of labels
    labelencode,    # encode a sequence of discrete values using a label map
    labeldecode,    # decode the label to the associated discrete value
    groupindices,   # grouped indices based on labels

    # crossval
    CrossValGenerator,  # abstract base class for all cross-validation plans
    Kfold,              # K-fold cross validation plan
    LOOCV,              # leave-one-out cross validation plan
    RandomSub,          # repetitive random subsampling cross validation

    cross_validate,     # perform cross-validation

    # perfeval
    ROCNums,        # A class to capture ROC numbers

    correctrate,    # compute correct rate of predictions
    errorrate,      # compute error rate of predictions
    counthits,      # count the number of hits
    hitrate,        # compute hit-rate of ranked lists at a specific rank
    hitrates,       # compute hit-rate of ranked lists at multiple ranks
    roc,            # compute roc numbers from predictions
    true_positive,      # number of true positives 
    true_negative,      # number of true negatives 
    false_positive,     # number of false positives
    false_negative,     # number of false negatives
    true_positive_rate,     # rate of true positives 
    true_negative_rate,     # rate of true negatives
    false_positive_rate,    # rate of false positives
    false_negative_rate,    # rate of false negatives
    recall,             # recall computed from ROCNums
    precision,          # precision computed from ROCNums
    f1score,            # F1-score computed from ROCNums

    # deviation
    sqL2dist,       # squared L2 distance between two arrays
    L2dist,         # L2 distance between two arrays
    L1dist,         # L1 distance between two arrays
    Linfdist,       # L-inf distance between two arrays
    gkldiv,         # (Generalized) Kullback-Leibler divergence between two vectors
    meanad,         # mean absolute deviation
    maxad,          # maximum absolute deviation
    msd,            # mean squared deviation
    rmsd,           # root mean squared deviation
    nrmsd,          # normalized rmsd
    psnr            # peak signal-to-noise ratio (in dB)

    # source files

    include("utils.jl")
    include("datapre.jl")
    include("classification.jl")
    include("crossval.jl")
    include("perfeval.jl")
    include("deviation.jl")
end

