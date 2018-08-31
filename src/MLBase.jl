module MLBase

    using Reexport
    using IterTools
    using Random
    @reexport using StatsBase

    import Base: length, show, keys, precision, length, getindex
    import Base: iterate
    import Base.Order: lt, Ordering, ForwardOrdering, ReverseOrdering, Forward, Reverse
    import StatsBase: RealVector, IntegerVector, RealMatrix, IntegerMatrix, RealArray
    import IterTools: product

    export

    # reexport from Base.Order
    Forward, Reverse,

    # utils
    repeach,        # repeat each element in a vector
    repeachcol,     # repeat each column in a matrix
    repeachrow,     # repeat each row in a matrix

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
    StratifiedKfold,        # stratified K-fold cross validation plan
    LOOCV,              # leave-one-out cross validation plan
    RandomSub,              # repeated random subsampling cross validation
    StratifiedRandomSub,    # stratified repeated random subsampling

    cross_validate,     # perform cross-validation

    # perfeval
    ROCNums,        # A class to capture ROC numbers

    correctrate,    # compute correct rate of predictions
    errorrate,      # compute error rate of predictions
    confusmat,      # compute confusion matrix
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

    # modeltune
    gridtune           # grid-based model tuning (search best config)

    # source files

    include("utils.jl")
    include("classification.jl")
    include("crossval.jl")
    include("perfeval.jl")
    include("modeltune.jl")

    include("deprecates.jl")
end
