
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

gt = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
pr = [0, 0, 1, 0, 0, 1, 1, 0, 1, 2, 2, 2, 2, 0, 1]

r0 = ROCNums{Int}(10, 5, 6, 4, 1, 2)
@test rocnums(gt, pr) == r0

gt = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
ss = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]

@test rocnums(gt, ss, 0.25) == ROCNums{Int}(6, 5, 6, 2, 3, 0)
@test rocnums(gt, ss, 0.55) == ROCNums{Int}(6, 5, 6, 5, 0, 0)
@test rocnums(gt, ss, 0.75) == ROCNums{Int}(6, 5, 4, 5, 0, 2)

gt = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
pr = [1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1]
ss = [0.2, 0.2, 0.2, 0.3, 0.3, 0.6, 0.6, 0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8]

@test rocnums(gt, pr, ss, 0.00) == ROCNums{Int}(10, 5, 8, 0, 5, 0)
@test rocnums(gt, pr, ss, 0.25) == ROCNums{Int}(10, 5, 8, 3, 2, 0)
@test rocnums(gt, pr, ss, 0.50) == ROCNums{Int}(10, 5, 8, 5, 0, 0)
@test rocnums(gt, pr, ss, 0.75) == ROCNums{Int}(10, 5, 4, 5, 0, 5)
@test rocnums(gt, pr, ss, 1.00) == ROCNums{Int}(10, 5, 0, 5, 0, 10)
