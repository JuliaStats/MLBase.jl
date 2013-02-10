# common stuff useful across the package

typealias RealVec Union(AbstractVector{Float32}, AbstractVector{Float64})
typealias RealMat Union(AbstractMatrix{Float32}, AbstractMatrix{Float64})
typealias RealVecOrMat Union(RealVec, RealMat)

# useful tools for testing & comparison

is_approx(x::Real, y::Real, tol::Real) = abs(x - y) < tol

function is_approx(x::Array, y::Array, tol::Real)
	size(x) == size(y) && all( abs(x - y) .< tol )
end
