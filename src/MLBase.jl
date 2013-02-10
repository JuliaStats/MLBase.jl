module MLBase

	export 
		# common
		is_approx,

		# prob_comp
		logsumexp, logsumexp!, softmax, softmax!, 
		
		# sampling_tools
		sample_by_weights


	include("common.jl")	
	include("prob_comp.jl")
	include("sampling_tools.jl")
	include("iter_process.jl")
end