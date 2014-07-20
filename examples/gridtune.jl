# A simple example to demonstrate the use of gridtune

using MLBase
using MultivariateStats

## prepare data

n_tr = 20  # number of training samples
n_te = 10  # number of testing samples
d = 5      # dimension of observations

theta = randn(d)
X_tr = randn(n_tr, d)
y_tr = X_tr * theta + 0.1 * randn(n_tr)
X_te = randn(n_te, d)
y_te = X_te * theta + 0.1 * randn(n_te)

## tune the model

function estfun(regcoef, bias)
    s = ridge(X_tr, y_tr, regcoef; bias=bias)
    return bias ? (s[1:end-1], s[end]) : (s, 0.0)
end

evalfun(m) = msd(X_te * m[1] + m[2], y_te) 

r = gridtune(estfun, evalfun, 
            ("regcoef", [1.0e-3, 1.0e-2, 1.0e-1, 1.0]), 
            ("bias", (true, false)); 
            ord=Reverse,
            verbose=true)

best_model, best_cfg, best_score = r

## print results

θ, b = best_model
println("Best model:") 
println("  θ = $(θ')"), 
println("  b = $b")
println("Best config: regcoef = $(best_cfg[1]), bias = $(best_cfg[2])")
println("Best score: $(best_score)")

