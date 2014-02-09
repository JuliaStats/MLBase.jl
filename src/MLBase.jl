module MLBase

    using ArrayViews
    using StatsBase

    import Base: length, show, keys, precision
    import StatsBase: RealVector, IntegerVector, RealMatrix, IntegerMatrix

    export

    # reexport some functions from StatsBase
    counts, addcounts!, countmap, proportions, 
    indicatormat, sample, weights, WeightVec, wmean,

    # utils
    repeach,        # repeat each element in a vector 
    repeachcol,     # repeat each column in a matrix
    repeachrow,     # repeat each row in a matrix
    counteq,        # count the number of equal pairs
    countne,        # count the number of non-equal pairs
        
    # classification
    ToMax,          # empty type to indicate the higher the value the better
    ToMin,          # empty type to indicate the lower to value the better
    ToMaxOrMin,     # Union(ToMax, ToMin)
    LabelMap,       # a type to represent a label map

    to_max,         # construct an instance of ToMax
    to_min,         # construct an instance of ToMin
    better,         # compare w.r.t. ToMax or ToMin
    classify,       # predict class label(s) based on score values
    classify!,      # inplace version of classify
    classify_withscore,     # classify with additional output of best score
    classify_withscores,    # classify with additional output of best scores
    classify_withscores!,   # inplace version of classify_withscores
    labelmap,       # construct a label map from a list of labels
    labelencode,    # encode a sequence of discrete values using a label map
    groupindices,   # grouped indices based on labels

    # perfeval
    ROCNums,        # A class to capture ROC numbers

    correctrate,    # compute correct rate of predictions
    errorrate,      # compute error rate of predictions
    rocnums,        # compute roc numbers from predictions (return ROCNums instance)
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
    f1score             # F1-score computed from ROCNums

    # source files

    include("utils.jl")
    include("classification.jl")
    include("perfeval.jl")
end
