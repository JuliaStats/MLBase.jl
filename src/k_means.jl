# K-means algorithm

require("Options")
using OptionsMod

using Distance

###########################################################
#
# 	K-means options
#
###########################################################

type KmeansOpts
	max_iters::Int
	tol::RealValue
	weights::Union(Nothing, RealVec)	
end

function kmeans_opts(opts::Options)
	@defaults opts max_iter=200 tol=1.0e-6 
	@defaults opts weights=nothing
	
	kopts = KmeansOpts(max_iter, tol, weights)
	
	@check_used opts
	return kopts
end


###########################################################
#
# 	Core implementation options
#
###########################################################

type KmeansState
	centers::RealMat,
	assignments::RealVec,
	costs::RealVec,
	counts::Vector{Int},
	dmat::RealMat
end

function _kmeans!(
	x::RealMat, 
	centers::RealMat, 
	assignments::RealVec,
	costs::RealVec,
	counts::Vector{Int},
	opts::KmeansOpts)
	
	iter_opts = iter_options(:minimize, opts.max_iters, opts.tol)
	
	
		
end