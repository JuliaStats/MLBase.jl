module MLBase

	export 
		# common
		is_approx, FPVec, FPMat, FPVecOrMat, 

		# prob_comp
		entropy, entropy!, logsumexp, logsumexp!, softmax, softmax!, 
		
		# sampling_tools
		sample_by_weights, sample_without_replacement

	include("common.jl")	
	include("prob_comp.jl")
	include("sampling_tools.jl")
end