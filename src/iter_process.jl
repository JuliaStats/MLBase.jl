# Driver functions to run an iterative process

###########################################################
#
#	options to control an iterative procedure
#
###########################################################

type IterOptions
	direction::Symbol
	max_iter::Int
	tol::Real
	
	function IterOptions(d::Symbol, m::Int, t::Real)
		if !(d == :dontcare || d == :maximize || d == :minimize)
			throw(ArgumentError("Invalid value for direction."))
		end
		if m < 1
			throw(ArgumentError("max_iter must be at least 1."))
		end
		if t <= 0
			throw(ArgumentError("tol must be positive."))
		end
		
	end
end

iter_options(dir::Symbol) = IterOptions(dir, 100, 1.0e-8)
iter_options(dir::Symbol, max_iter::Int, tol::Real) = IterOptions(dir, max_iter, tol)

abstract IterMonitor

###########################################################
#
#	driver function
#
###########################################################

function test_convergence(dir::Symbol, ch::Real, tol::Real)
	if dir == :maximize
		if ch < -tol
			warn("Objective value changes towards the opposite direction.")
		end
	elseif dir == :minimize
		if ch > tol
			warn("Objective value changes towards the opposite direction.")
		end
	end
	
	return abs(ch) < tol
end


function iterative_update!(pb, state, opts::IterOptions, monitor::Union(Nothing, IterMonitor))
	
	t = 0
	converged = false
	objv = nothing
	
	use_mon = !isa(monitor, Nothing)
	
	if use_mon
		on_start(monitor, pb, state)
	end
	
	while !converged && t < opts.max_iter
		t += 1
		
		update!(pb, state)		
		
		pre_objv = objv
		objv = evaluate_objv(pb, state)
		
		if t > 1
			objv_change = objv - pre_objv
			converged = test_convergence(opts.direction, objv_change, opts.tol)
		else
			objv_change = nothing
		end
		
		if use_mon
			on_updated(monitor, pb, state, t, objv, objv_change)
		end
	end
	
	if use_mon
		on_finished(monitor, pb, state, t, objv, converged)
	end
end


###########################################################
#
#	monitors
#
###########################################################

type StdIterationMonitor{Level}
end

on_start(mon::IterMonitor, pb, state) = nothing
on_updated(mon::IterMonitor, pb, state, t, objv, change) = nothing
on_finished(mon::IterMonitor, pb, state, t, objv, converged) = nothing

function stdmon_on_finished(t::Int, objv::Real, converged::Bool)
	if converged
		println("Converged with $t iterations (objv = $objv).")
	else
		println("Terminated after $t iterations without convergence (objv = $objv).")
	end
end


on_finish(::StdIterationMonitor{:final}, pb, state, t, objv, converged) = stdmon_on_finished(t, objv, converged)
on_finish(::StdIterationMonitor{:iter}, pb, state, t, objv, converged) = stdmon_on_finished(t, objv, converged)

function on_finish(::StdIterationMonitor{:iter}, pb, state, t, objv, change)
	if change == nothing
		@printf "Iter %5d:  %12.6e" t objv
	else
		@printf "Iter %5d:  %12.6e  %12.6e" t objv change
	end
end


function get_std_iter_monitor(level::Symbol)
	if level == :none
		nothing
	elseif level == :final || level == :iter
		StdIterationMonitor{level}()
	else
		error("Invalid monitor level: $level")
	end
end
