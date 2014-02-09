
using MLBase
using Base.Test

import StatsBase
import StatsBase: harmmean 

## correctrate & errorrate

a = [1, 1, 1, 2, 2, 2, 3, 3]
b = [1, 1, 2, 2, 2, 3, 3, 3]

@test correctrate(a, b) == 0.75
@test errorrate(a, b) == 0.25

## ROCNums

r = ROCNums{Int}(
    100, # p == tp + fn
    200, # n == tn + fp
    80,  # tp
    150, # tn
    50,  # fp
    20)  # fn

@test true_positive(r) == 80
@test true_negative(r) == 150
@test false_positive(r) == 50
@test false_negative(r) == 20

@test true_positive_rate(r) == 0.80
@test true_negative_rate(r) == 0.75
@test false_positive_rate(r) == 0.25
@test false_negative_rate(r) == 0.20

@test recall(r) == 0.80
@test precision(r) == (8/13)
@test_approx_eq f1score(r) harmmean([recall(r), precision(r)])
