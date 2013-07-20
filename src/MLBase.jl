module MLBase

    export
        @check_argdims,

        # intstats
        add_icounts!, icounts, add_icounts2!, icounts2,
        add_wcounts!, wcounts, add_wcounts2!, wcounts2,
        sort_indices, sorted_indices_to_groups, group_indices,
        repeat_eachelem,

        # options
        VERBOSE_NONE, VERBOSE_PROC, VERBOSE_ITER, VERBOSE_STEP, VERBOSE_DETAIL,
        verbosity_level

    # common tools

    macro check_argdims(cond)
        :( if !($(esc(cond)))
            throw(ArgumentError("Invalid argument dimensions.")) 
        end)  
    end

    # components

    include("intstats.jl")

end
