# Useful tools for sampling

###########################################################
#
#   A fast sampling method to sample x ~ p,
#   where p is proportional to given weights.
#
#   This algorithm performs the sampling without
#   computing p (normalizing the weights).
#
###########################################################

function sample_by_weights(w::Vector{Float64}, totalw::Float64)
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

sample_by_weights(w::Vector{Float64}) = sample_by_weights(w, sum(w))
