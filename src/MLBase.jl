module MLBase

    using ArrayViews

    export

    # utils
    repeach, repeachcol, repeachrow,

    # iteroptim
    IterOptimProblem, IterOptimMonitor, IterOptimInfo, iter_optim!, 
    on_initialized, on_iteration, on_finished,
    objective, update!, initialize, solve,
    VERBOSE_NONE, VERBOSE_PROC, VERBOSE_ITER, VERBOSE_STEP, VERBOSE_DETAIL,
    verbosity_level
        

    # components

    include("utils.jl")
    include("labelmani.jl")
    include("iteroptim.jl")

end
