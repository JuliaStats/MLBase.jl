module MLBase

    using ArrayViews
    using StatsBase

    import Base: length, show, keys

    export

    # reexport some functions from StatsBase
    counts, accounts!, countmap, indicatormat, 
    sample, weights, WeightVec, wmean,

    # utils
    repeach, repeachcol, repeachrow,
        
    # labelmani
    LabelMap, labelmap, labelencode, groupindices

    # source files

    include("utils.jl")
    include("labelmani.jl")
end
