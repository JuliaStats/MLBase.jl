module MLBase

    import Base.show, Base.logdet, Base.full, Base.inv, Base.+, Base.*, Base./
    import Base.LinAlg.BLAS.axpy!, Base.LinAlg.BLAS.gemv!, Base.LinAlg.BLAS.gemv
    import Base.LinAlg.BLAS.gemm!, Base.LinAlg.BLAS.gemm

    export
        # vec_arith
        @check_argdims,
        add!, add_cols!, add_cols, add_rows!, add_rows, 
        sub!, sub_cols!, sub_cols, sub_rows!, sub_rows, 
        mul!, mul_cols!, mul_cols, mul_rows!, mul_rows,
        add_diag!, add_diag,

        # vecreduc
        vsum!, vsum, vmean!, vmean, vmax!, vmax, vmin!, vmin,
        vasum!, vasum, vamax!, vamax, vamin!, vamin, 
        vsqsum!, vsqsum, vpowsum!, vpowsum,
        vdot!, vdot, vsqdiffsum!, vsqdiffsum,

        # norms
        vnorm!, vnorm, vdiffnorm!, vdiffnorm        


    # common tools

    macro check_argdims(cond)
        :( if !($(esc(cond)))
            throw(ArgumentError("Invalid argument dimensions.")) 
        end)  
    end

    # components

    include("vecarith.jl")
    include("vecreduc.jl")
    include("norms.jl")

end
