# Facilities for implementing iterative optimization algorithm

import Base.solve

abstract IterOptimProblem
abstract IterOptimSolution

function update!(prb::IterOptimProblem, sol::IterOptimSolution)
	error("update! has not been defined on ($(typeof(prb)), $(typeof(sol)))")
end

function objective(prb::IterOptimProblem, sol::IterOptimSolution)
	error("objective has not been defined on ($(typeof(prb)), $(typeof(sol)))")
end

function initialize(prb::IterOptimProblem)
	error("initialize has not been defined on $(typeof(prb))")
end


# Verbosity control options

const VERBOSE_NONE = 0     # display nothing
const VERBOSE_PROC = 1     # display at begining/end of the entire procedure
const VERBOSE_ITER = 2     # display at each iteration
const VERBOSE_STEP = 3     # display at each step
const VERBOSE_DETAIL = 4   # display lots of details

function verbosity_level(sym::Symbol; 
	errmsg::ASCIIString="Invalid value for the display option.")

	sym == :off || sym == :none ? VERBOSE_NONE :
	sym == :proc ? VERBOSE_PROC :
	sym == :iter ? VERBOSE_ITER :
	sym == :step ? VERBOSE_STEP :
	sym == :detail ? VERBOSE_DETAIL : 
	throw(ArgumentError(errmsg))
end

# Procedure control options

immutable IterOptimOptions
	maxiter::Int
	tol::Float64
	display::Symbol
	verbose::Int

	function IterOptimOptions(;maxiter::Integer=100, tol::Real=1.0e-8, display::Symbol=:none)
		if maxiter <= 0
			throw(ArgumentError("IterOptimOptions: maxiter must be a positive integer."))
		end

		if tol <= 0.
			throw(ArgumentError("IterOptimOptions: tol must be a postivie real value."))
		end

		new(int(maxiter), float64(tol), display, verbosity_level(display))
	end
end

# Procedure skeleton

immutable IterOptimInfo
	objective::Float64
	converged::Bool
	niters::Int

	function IterOptimInfo(objv::Real, converged::Bool, nitrs::Integer)
		new(float64(objv), converged, int(nitrs))
	end
end

function iter_optim!(problem::IterOptimProblem, solution::IterOptimSolution, opts::IterOptimOptions)

	# get options
	maxiter = opts.maxiter
	tol = opts.tol
	verbose = opts.verbose

	# preamble

	converged::Bool = false
	it::Int = 0
	objv = objective(problem, solution)

	if verbose >= VERBOSE_PROC
		@printf("%6s    %12s     %12s\n", "Iter", "Objective", "Objv.change")
		println("------------------------------------------")
		@printf("%6d    %12.4e\n", 0, objv)
	end

	# main loop

	while !converged && it < maxiter
		it += 1

		# perform update (inplace)
		update!(problem, solution)

		# decide convergence
		objv_pre = objv
		objv = objective(problem, solution)
		converged = abs(objv - objv_pre) < tol

		if verbose >= VERBOSE_ITER
			@printf("%6d    %12.4e     %12.4e\n", it, objv, objv - objv_pre)
		end
	end

	if verbose >= VERBOSE_PROC
		if converged
			println("Iterative optimization converged with $it iterations.")
		else
			println("Iterative optimization terminated after $it iterations without convergence.")
		end
	end

	return IterOptimInfo(objv, converged, it)
end


function solve(prb::IterOptimProblem, opts::IterOptimOptions)
	sol = initialize(prb)
	info = iter_optim!(prb, sol, opts)
	return (sol, info)
end




