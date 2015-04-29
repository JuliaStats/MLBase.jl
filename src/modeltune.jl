# Model tuning

_always_true(xs...) = true

function gridtune(estfun::Function,                          # model estimation function
                  evalfun::Function,                         # model evaluation function
                  params::@compat(Tuple{String, Any})...;    # parameters to tune
                  ord::Ordering=Forward,                     # ordering of score
                  verbose::Bool=false)                       # whether to display the progress      

    np = length(params)
    pnames = [p[1] for p in params]
    pvals = [p[2] for p in params]

    # main loop
    t = 0
    first = true
    local best_score, best_model, best_cfg
    for cf in product(pvals...)
        t += 1
        m = estfun(cf...)
        if m == nothing
            continue
        end
        v = evalfun(m)
        update_best = (first || lt(ord, best_score, v))
        first = false

        if update_best
            best_model = m
            best_cfg = cf
            best_score = v
        end

        if verbose
            print("[")
            for i = 1:np
                print("$(pnames[i])=$(cf[i])")
                if i < np 
                    print(", ") 
                end
            end
            println("] => $v")
        end
    end

    if first
        error("None of the configs are valid.")
    end
    return (best_model, best_cfg, best_score)
end

