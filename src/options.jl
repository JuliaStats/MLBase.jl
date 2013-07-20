# Helpers for algorithm options


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
