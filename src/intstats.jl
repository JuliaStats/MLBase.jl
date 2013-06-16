# Statistics related to integers

#################################################
#
#	Integer counting
#
#################################################

function add_icounts!{T<:Integer}(r::Array, v::Array{T})
	for i in 1 : length(v)
		r[v[i]] += 1
	end
	r
end

icounts{T<:Integer}(k::Integer, v::Array{T}) = add_icounts!(zeros(Int, k), v)

function add_icounts2!{T<:Integer}(r::Matrix, v1::Array{T}, v2::Array{T})
	@check_argdims length(v1) == length(v2)
	for i in 1 : length(v1)
		r[v1[i], v2[i]] += 1
	end
	r
end

function icounts2{T<:Integer}(k1::Integer, k2::Integer, v1::Array{T}, v2::Array{T})
	add_icounts2!(zeros(Int, k1, k2), v1, v2)
end

function add_wcounts!{T<:Integer, W<:Real}(r::Array, v::Array{T}, w::Array{W})
	@check_argdims length(v) == length(w)
	for i in 1 : length(v)
		r[v[i]] += w[i]
	end
	r
end

wcounts{T<:Integer, W<:Real}(k::Integer, v::Array{T}, w::Array{W}) = add_wcounts!(zeros(W, k), v, w)

function add_wcounts2!{T<:Integer, W<:Real}(r::Matrix, v1::Array{T}, v2::Array{T}, w::Array{W})
	@check_argdims length(v1) == length(v2) == length(w)
	for i in 1 : length(v1)
		r[v1[i], v2[i]] += w[i]
	end
	r
end

function wcounts2{T<:Integer, W<:Real}(k1::Integer, k2::Integer, v1::Array{T}, v2::Array{T}, w::Array{W})
	add_wcounts2!(zeros(W, k1, k2), v1, v2, w)
end


#################################################
#
#	Indice arrangement
#
#################################################

function sort_indices{T<:Integer}(k::Integer, v::Array{T})
	n = length(v)

	# first pass: counting
	cnts::Vector{Int} = icounts(k, v)

	offsets = Array(Int, k)
	offsets[1] = 0
	for i in 2 : k
		offsets[i] = offsets[i-1] + cnts[i-1]
	end

	# second pass: sort indices
	z = Array(Int, n)
	for i in 1 : n
		z[offsets[v[i]] += 1] = i
	end

	return (z, cnts)
end


function sorted_indices_to_groups{T<:Integer}(sinds::Vector{T}, cnts::Vector{Int})
	k = length(cnts)
	p = 0
	grps = Array(Vector{Int}, k)
	for i in 1 : k
		ci = cnts[i]
		grps[i] = sinds[p+1 : p+ci]
		p += ci
	end
	grps
end

function group_indices{T<:Integer}(k::Integer, v::Array{T})
	z::Vector{Int}, cnts::Vector{Int} = sort_indices(k, v)
	sorted_indices_to_groups(z, cnts)
end

function repeat_eachelem{T, N<:Integer}(x::AbstractArray{T}, cnts::Vector{N})
	@check_argdims length(x) == length(cnts)
	r = Array(T, sum(cnts))
	p = 0
	for i in 1 : length(x)
		ci = cnts[i]
		r[p+1:p+ci] = x[i]
		p += ci
	end
	r
end

function repeat_eachelem{T}(x::AbstractArray{T}, n::Integer)
	r = Array(T, n * length(x))
	p = 0
	for i in 1 : length(x)
		r[p+1:p+n] = x[i]
		p += n
	end
	r
end




