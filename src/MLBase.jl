module MLBase

    export
        # prob_comp
        entropy, entropy!, logsumexp, logsumexp!, softmax, softmax!,

        # sampling_tools
        sample_by_weights, sample_without_replacement

    include("prob_comp.jl")
    include("sampling_tools.jl")
end
