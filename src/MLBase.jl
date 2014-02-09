module MLBase

    using ArrayViews
    using StatsBase

    import Base: length, show, keys
    import StatsBase: RealVector, IntegerVector, RealMatrix, IntegerMatrix

    export

    # reexport some functions from StatsBase
    counts, accounts!, countmap, proportions, 
    indicatormat, sample, weights, WeightVec, wmean,

    # utils
    repeach, repeachcol, repeachrow,
    counteq, countne, 
        
    # classification
    classify, classify!, to_max, to_min, ToMax, ToMin, ToMaxOrMin,
    classify_withscore, classify_withscores, classify_withscores!,
    LabelMap, labelmap, labelencode, groupindices,

    # perfeval
    correctrate, errorrate

    # source files

    include("utils.jl")
    include("classification.jl")
    include("perfeval.jl")
end
