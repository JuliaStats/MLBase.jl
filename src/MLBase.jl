module MLBase

    export
        # vec_arith
        @check_argdims,
        add!, add_cols!, add_cols, add_rows!, add_rows, 
        sub!, sub_cols!, sub_cols, sub_rows!, sub_rows, 
        mul!, mul_cols!, mul_cols, mul_rows!, mul_rows,

        # vecreduc
        vsum!, vsum, vmean!, vmean, vmax!, vmax, vmin!, vmin,
        vasum!, vasum, vamax!, vamax, vamin!, vamin, vsqsum!, vsqsum,
        
        # sampling_tools
        sample_by_weights, sample_without_replacement


    # common tools

    macro check_argdims(cond)
        :( if !($(esc(cond)))
            throw(ArgumentError("Invalid argument dimensions.")) 
        end)  
    end

    # components

    include("vecarith.jl")
    include("vecreduc.jl")

end
