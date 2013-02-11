# Useful tools for sampling

###########################################################
#
# 	A fast sampling method to sample x ~ p,
# 	where p is proportional to given weights.
#
# 	This algorithm performs the sampling without 
#	computing p (normalizing the weights).
#
###########################################################

function sample_by_weights(w::RealVec, totalw::FloatingPoint)
	n = length(w)
	t = rand() * totalw
	
	x = 1
	s = w[1]
	
	while x < n && s < t
		x += 1
		s += w[x]
	end
	return x
end

sample_by_weights(w::RealVec) = sample_by_weights(w, sum(w))


function partial_shuffle!(r::Vector, k::Int)
	# randomly swapping k elements to front
	
	n = length(r)
	for i = 1 : k
		j = rand(i:n)
		if j != i
			t = r[i]
			r[i] = r[j]
			r[j] = t
		end
	end
end


function sample_without_replacement(x::AbstractVector, k::Int)
	n = length(x)
	if !(k >= 0 && k <= n)
		throw(ArgumentError("k must be in [0, length(x)]."))
	end
	r = vec(x)
	
	partial_shuffle!(r, k)
	return r[1:k]
end


