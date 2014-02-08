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
        
    # classification
    LabelMap, labelmap, labelencode, groupindices

    # source files

    include("utils.jl")
    include("classification.jl")
end
