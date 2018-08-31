using MLBase
using Test

## gridtune

oracle = Dict{Tuple{Int, Symbol},Float64}(
    (1, :a) => 2.0,
    (2, :a) => 1.0,
    (3, :a) => 3.0,
    (1, :b) => 4.0,
    (3, :b) => 2.5
)

estfun = (x, y) -> get(oracle, (x, y), nothing)
evalfun = v -> 2 * v
p1 = ("x", 1:3)
p2 = ("y", [:a, :b])

(rmodel, rcfg, rscore) = gridtune(estfun, evalfun, p1, p2)
@test rmodel == 4.0
@test rcfg == (1, :b)
@test rscore == 8.0

(rmodel, rcfg, rscore) = gridtune(estfun, evalfun, p1, p2; ord=Reverse)
@test rmodel == 1.0
@test rcfg == (2, :a)
@test rscore == 2.0
