# Computation of norms

function vnorm!(r::Array, x::Matrix, p::Real, dim::Integer)
	if !(p > 0)
		throw(ArgumentError("p must be a positive value"))
	end

	if p == 1
		vasum!(r, x, dim)
	elseif p == 2
		vsqsum!(r, x, dim)
		for i in 1 : length(r)
			r[i] = sqrt(r[i])
		end
	elseif isinf(p)
		vamax!(r, x, dim)
	else
		vpowsum!(r, x, p, dim)
		inv_p = inv(p)
		for i in 1 : length(r)
			r[i] .^= inv_p
		end
	end
	r
end

function vnorm(x::Matrix, p::Real, dim::Integer)
	rlen = _reduc_length(size(x, 1), size(x, 2), dim)
	r = Array(eltype(x), rlen)
	vnorm!(r, x, p, dim)
end


# norm of difference

type L1diffReduc <: AbstractReduction end

result_type{T<:Number}(op::L1diffReduc, ty1::Type{T}, ty2::Type{T}) = T
empty_value{T<:Number}(op::L1diffReduc, ty1::Type{T}, ty2::Type{T}) = zero(T)
init_value{T<:Number}(op::L1diffReduc, x::T, y::T) = abs(x - y)
combine_value{T<:Number}(op::L1diffReduc, s::T, x::T, y::T) = s + abs(x - y)

type LinfdiffReduc <: AbstractReduction end

result_type{T<:Number}(op::LinfdiffReduc, ty1::Type{T}, ty2::Type{T}) = T
empty_value{T<:Number}(op::LinfdiffReduc, ty1::Type{T}, ty2::Type{T}) = zero(T)
init_value{T<:Number}(op::LinfdiffReduc, x::T, y::T) = abs(x - y)
combine_value{T<:Number}(op::LinfdiffReduc, s::T, x::T, y::T) = (a = abs(x - y); s > a ? s : a)

immutable LpdiffReduc{T<:Real} <: AbstractReduction 
	p::T
end

result_type{T<:Number}(op::LpdiffReduc, ty1::Type{T}, ty2::Type{T}) = T
empty_value{T<:Number}(op::LpdiffReduc, ty1::Type{T}, ty2::Type{T}) = zero(T)
init_value{T<:Number}(op::LpdiffReduc, x::T, y::T) = abs(x - y) .^ op.p
combine_value{T<:Number}(op::LpdiffReduc, s::T, x::T, y::T) = s + abs(x - y) .^ op.p

function vdiffnorm!(r::Array, x::Matrix, y::Matrix, p::Real, dim::Integer)
	if !(p > 0)
		throw(ArgumentError("p must be a positive value"))
	end

	if p == 1
		vreduce!(r, L1diffReduc(), x, y, dim)
	elseif p == 2
		vsqdiffsum!(r, x, y, dim)
		for i in 1 : length(r)
			r[i] = sqrt(r[i])
		end
	elseif isinf(p)
		vreduce!(r, LinfdiffReduc(), x, y, dim)
	else
		vreduce!(r, LpdiffReduc(p), x, y, dim)
		inv_p = inv(p)
		for i in 1 : length(r)
			r[i] .^= inv_p
		end
	end
	r
end

function vdiffnorm(x::Matrix, y::Matrix, p::Real, dim::Integer)
	rlen = _reduc_length(size(x, 1), size(x, 2), dim)
	r = Array(promote_type(eltype(x), eltype(y)), rlen)
	vdiffnorm!(r, x, y, p, dim)
end


# normalize

normalize!(x::Vector, p::Real) = mul!(x, inv(norm(x, p)))
normalize(x::Vector, p::Real) = x * inv(norm(x, p))

function normalize!(x::Matrix, p::Real, dim::Integer)
	if dim == 1
		mul_rows!(x, rcp!(vnorm(x, p, 1)))
	elseif dim == 2
		mul_cols!(x, rcp!(vnorm(x, p, 2)))
	else
		throw(ArgumentError("The value of dim must be either 1 or 2."))
	end
end

function normalize(x::Matrix, p::Real, dim::Integer)
	if dim == 1
		mul_rows(x, rcp!(vnorm(x, p, 1)))
	elseif dim == 2
		mul_cols(x, rcp!(vnorm(x, p, 2)))
	else
		throw(ArgumentError("The value of dim must be either 1 or 2."))
	end	
end


