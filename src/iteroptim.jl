# Facilities for implementing iterative optimization algorithm

import Base.solve

abstract IterOptimProblem
abstract IterOptimMonitor

function update!(prb::IterOptimProblem, sol)
	error("update! has not been defined on ($(typeof(prb)), $(typeof(sol)))")
end

function objective(prb::IterOptimProblem, sol)
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

# Procedure skeleton

immutable IterOptimInfo
	objective::Float64
	converged::Bool
	niters::Int

	function IterOptimInfo(objv::Real, converged::Bool, nitrs::Integer)
		new(float64(objv), converged, int(nitrs))
	end
end


function iter_optim!(problem::IterOptimProblem, state, maxiter::Integer, tol::Real)
	# preamble

	converged::Bool = false
	it::Int = 0
	objv = objective(problem, state)

	# main loop

	while !converged && it < maxiter
		it += 1

		# perform update (inplace)
		update!(problem, state)

		# decide convergence
		objv_pre = objv
		objv = objective(problem, state)
		converged = abs(objv - objv_pre) < tol
	end

	return IterOptimInfo(objv, converged, it)
end

function iter_optim!(problem::IterOptimProblem, state, maxiter::Integer, tol::Real, monitor::IterOptimMonitor)
	# preamble

	converged::Bool = false
	it::Int = 0
	objv = objective(problem, state)

	on_initialized(monitor, objv)

	# main loop

	while !converged && it < maxiter
		it += 1

		# perform update (inplace)
		update!(problem, state)

		# decide convergence
		objv_pre = objv
		objv = objective(problem, state)
		converged = abs(objv - objv_pre) < tol

		on_iteration(monitor, it, objv, objv - objv_pre)
	end

	on_finished(monitor, it, objv, converged)

	return IterOptimInfo(objv, converged, it)
end


# standard monitor

immutable StdIterOptimMonitor <: IterOptimMonitor
	verbose::Int

	StdIterOptimMonitor(verbose::Int) = new(verbose)
	StdIterOptimMonitor(display::Symbol) = new(verbosity_level(display))
end

function on_initialized(m::StdIterOptimMonitor, objv::Real)
	if m.verbose >= VERBOSE_PROC
		@printf("%6s    %12s     %12s\n", "Iter", "Objective", "Objv.change")
		println("------------------------------------------")
		@printf("%6d    %12.4e\n", 0, objv)
	end	
end

function on_iteration(m::StdIterOptimMonitor, it::Integer, objv::Real, objv_ch::Real)
	if m.verbose >= VERBOSE_ITER
		@printf("%6d    %12.4e     %12.4e\n", it, objv, objv_ch)
	end	
end

function on_finished(m::StdIterOptimMonitor, it::Integer, objv::Real, converged::Bool)
	if m.verbose >= VERBOSE_PROC
		if converged
			println("Iterative optimization converged with $it iterations.")
		else
			println("Iterative optimization terminated after $it iterations without convergence.")
		end
	end	
end


# convenient functions

function iter_optim!(problem::IterOptimProblem, state, maxiter::Integer, tol::Real, disp::Symbol)
	mon = StdIterOptimMonitor(verbosity_level(disp))
	iter_optim!(problem, state, maxiter, tol, mon)
end


function solve(prb::IterOptimProblem, maxiter::Integer, tol::Real, mon::IterOptimMonitor)
	sol = initialize(prb)
	info = iter_optim!(prb, sol, maxiter, tol, mon)
	return (sol, info)
end

function solve(prb::IterOptimProblem, maxiter::Integer, tol::Real, disp::Symbol)
	sol = initialize(prb)
	info = iter_optim!(prb, sol, maxiter, tol, disp)
	return (sol, info)
end

