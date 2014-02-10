# compute deviation between two arrays in different ways

# squared L2 distance
function sqL2dist{T<:Number}(a::ContiguousArray{T}, b::ContiguousArray{T})
    n = length(a)
    length(b) == n || throw(DimensionMismatch("Input dimension mismatch"))
    r = 0.0
    for i = 1:n
        @inbounds r += abs2(a[i] - b[i]) 
    end
    return r
end

# L2 distance
L2dist{T<:Number}(a::ContiguousArray{T}, b::ContiguousArray{T}) = sqrt(sqL2dist(a, b))

# L1 distance
function L1dist{T<:Number}(a::ContiguousArray{T}, b::ContiguousArray{T})
    n = length(a)
    length(b) == n || throw(DimensionMismatch("Input dimension mismatch"))
    r = 0.0
    for i = 1:n
        @inbounds r += abs(a[i] - b[i]) 
    end
    return r    
end

# Linf distance
function Linfdist{T<:Number}(a::ContiguousArray{T}, b::ContiguousArray{T})
    n = length(a)
    length(b) == n || throw(DimensionMismatch("Input dimension mismatch"))
    r = 0.0
    for i = 1:n
        @inbounds v = abs(a[i] - b[i]) 
        if r < v
            r = v
        end
    end
    return r     
end

# Generalized KL-divergence
function gkldiv{T<:FloatingPoint}(a::ContiguousArray{T}, b::ContiguousArray{T})
    n = length(a)
    r = 0.0
    for i = 1:n
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        if ai > 0
            r += (ai * log(ai / bi) - ai + bi)
        else
            r += bi
        end
    end
    return r::Float64
end

# MeanAD: mean absolute deviation
meanad{T<:Number}(a::ContiguousArray{T}, b::ContiguousArray{T}) = L1dist(a, b) / length(a)

# MaxAD: maximum absolute deviation
maxad{T<:Number}(a::ContiguousArray{T}, b::ContiguousArray{T}) = Linfdist(a, b)

# MSD: mean squared deviation
msd{T<:Number}(a::ContiguousArray{T}, b::ContiguousArray{T}) = sqL2dist(a, b) / length(a)

# RMSD: root mean squared deviation
rmsd{T<:Number}(a::ContiguousArray{T}, b::ContiguousArray{T}) = sqrt(msd(a, b))

# NRMSD: normalized mean squared deviation
function nrmsd{T<:Number}(a::ContiguousArray{T}, b::ContiguousArray{T})
    amin, amax = extrema(a)
    rmsd(a, b) / (amax - amin)
end

# PSNR: peak signal-to-noise ratio
function psnr{T<:Real}(a::ContiguousArray{T}, b::ContiguousArray{T}, maxv::Real)
    20. * log10(maxv) - 10. * log10(msd(a, b))
end
