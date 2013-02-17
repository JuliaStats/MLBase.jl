# Tools for probability computation

using Devectorize

# a numerically stable method to compute
#
#	log( sum_i exp(x_i) )
#
function logsumexp(x::F64Vec)
	mx = max(x)
	n = length(x)
	s = 0.
	for i = 1 : n
		s += exp(x[i] - mx) 
	end
	log(s) + mx
end

function logsumexp!(r::F64Arr, x::F64Mat, dim::Int)
	if dim == 1
		if length(r) != size(x, 2)
			throw(ArgumentError("The length or must match the number of columns in x."))
		end
		m, n = size(x)
		
		@devec r[:] = max(x, (), 1)
		for j = 1 : n
			s = 0.
			mx = r[j]
			for i = 1 : m	
				s += exp(x[i,j] - mx)
			end
			r[j] = log(s) + mx
		end
		
	elseif dim == 2
		if length(r) != size(x, 1)
			throw(ArgumentError("The length or must match the number of rows in x."))
		end
		m, n = size(x)
		
		@devec r[:] = max(x, (), 2)
		s = zeros(m)
		for j = 1 : n
			for i = 1 : m
				s[i] += exp(x[i,j] - r[i])
			end
		end
		@devec r[:] = r + log(s)
		
	else
		throw(ArgumentError("dim must be either 1 or 2."))
	end
end


function logsumexp(x::F64Mat, dim::Int)
	if dim == 1
		r = zeros(1, size(x, 2))
		logsumexp!(r, x, dim)
	elseif dim == 2
		r = zeros(size(x, 1), 1)
		logsumexp!(r, x, dim)
	else
		throw(ArgumentError("dim must be either 1 or 2."))
	end
	return r
end



# numerical stable method to compute softmax
#
#	r[i] = exp(x[i]) / sum(exp(x))
#

function softmax!(r::F64Vec, x::F64Vec)
	if length(r) != length(x)
		throw(ArgumentError("The lengths of r and x must match."))
	end
	n = length(x)
	mx = max(x)
	
	# must use double as accumulator, even x is single 
	# otherwise, errors can build up very fast
	s = 0.0 
	for i = 1 : n
		r[i] = exp(x[i] - mx)
		s += r[i]
	end
	inv_s = 1/s
	
	@devec r[:] .*= inv_s
end

function softmax(x::F64Vec)
	r = similar(x)
	softmax!(r, x)
	return r
end

function softmax!(r::F64Mat, x::F64Mat, dim::Int)
	if !(dim == 1 || dim == 2)
		throw(ArgumentError("dim must be either 1 or 2."))
	end
	if size(r) != size(x)
		throw(ArgumentError("The sizes of r and x must match."))
	end
	m, n = size(x)
	
	if dim == 1 # by columns
		@devec mx = max(x, (), 1)
		for j = 1 : n
			s = 0.0
			for i = 1 : m
				s += (r[i,j] = exp(x[i,j] - mx[j]))
			end
			inv_s = rcp(s)
			@devec r[:,j] .*= inv_s
		end
	else
		# to make it cache-friendly, the structure is different
		@devec mx = max(x, (), 2)
		s = zeros(m)
		for j = 1 : n
			for i = 1 : m
				s[i] += (r[i,j] = exp(x[i,j] - mx[i]))
			end
		end
		@devec inv_s = rcp(s)
		
		for j = 1 : n
			@devec r[:,j] .*= inv_s
		end
	end
end


function softmax(x::F64Mat, dim::Int)
	r = similar(x)
	softmax!(r, x, dim)
	return r
end
	
