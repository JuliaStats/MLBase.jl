module MLBase

    # import of base functions to be extended
    import Base.show, Base.logdet, Base.full, Base.inv, Base.+, Base.*, Base.\, Base./

    # import of useful BLAS & LAPACK routines
    import Base.LinAlg.BLAS.axpy!, Base.LinAlg.BLAS.nrm2
    import Base.LinAlg.BLAS.gemv!, Base.LinAlg.BLAS.gemv
    import Base.LinAlg.BLAS.gemm!, Base.LinAlg.BLAS.gemm    
    import Base.LinAlg.BLAS.trmv!, Base.LinAlg.BLAS.trmv
    import Base.LinAlg.BLAS.trmm!, Base.LinAlg.BLAS.trmm
    import Base.LinAlg.LAPACK.trtrs! 

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
        vsqsum!, vsqsum, weighted_sqsum, vpowsum!, vpowsum,
        vdot!, vdot, vsqdiffsum!, vsqdiffsum, 

        # norms
        vnorm!, vnorm, vdiffnorm!, vdiffnorm,        

        # pdmat
        AbstractPDMat, PDMat, PDiagMat, ScalMat, 
        dim, full, whiten, whiten!, unwhiten, unwhiten!, add_scal!, add_scal,
        quad, quad!, invquad, invquad!, X_A_Xt, Xt_A_X, X_invA_Xt, Xt_invA_X


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
    include("pdmat.jl")

end
