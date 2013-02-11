# common types

typealias FPArr Union(AbstractArray{Float32}, AbstractArray{Float64})
typealias FPVec Union(AbstractVector{Float32}, AbstractVector{Float64})
typealias FPMat Union(AbstractMatrix{Float32}, AbstractMatrix{Float64})
typealias FPVecOrMat Union(FPVec, FPMat)

# useful tools for testing & comparison

is_approx(x::Real, y::Real, tol::Real) = abs(x - y) < tol

function is_approx(x::Array, y::Array, tol::Real)
	size(x) == size(y) && all( abs(x - y) .< tol )
end
