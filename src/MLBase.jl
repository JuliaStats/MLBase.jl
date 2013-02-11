module MLBase

	export 
		# common
		is_approx,

		# prob_comp
		logsumexp, logsumexp!, softmax, softmax!, 
		
		# sampling_tools
		sample_by_weights, sample_without_replacement


	include("common.jl")	
	include("prob_comp.jl")
	include("sampling_tools.jl")
	include("iter_process.jl")
end