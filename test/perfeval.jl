
using MLBase
using Base.Test

a = [1, 1, 1, 2, 2, 2, 3, 3]
b = [1, 1, 2, 2, 2, 3, 3, 3]

@test correctrate(a, b) == 0.75
@test errorrate(a, b) == 0.25

