module MLBase

	export 
		# common
		F64Arr, F64Vec, F64Mat, IntArr, IntVec, IntMat, 

		# prob_comp
		logsumexp, logsumexp!, softmax, softmax!, 
		
		# sampling_tools
		sample_by_weights, sample_without_replacement

	include("common.jl")	
	include("prob_comp.jl")
	include("sampling_tools.jl")
end