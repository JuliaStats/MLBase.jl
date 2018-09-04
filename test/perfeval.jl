
using MLBase
using Test

import StatsBase
import StatsBase: harmmean

## correctrate & errorrate

a = [0, 1, 1, 2, 2, 2, 3, 3]
b = [1, 1, 2, 2, 2, 3, 3, 3]

@test correctrate(a, b) == 0.625
@test errorrate(a, b) == 0.375

## confusmat

@test confusmat(a, b) == [0 1 0 0; 0 1 1 0; 0 0 2 1; 0 0 0 2]

## counthits & hitrates

gt = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
rs = [1 2 2 1 3 2 1 1 3 3;
      2 3 1 3 1 1 2 2 1 2;
      4 4 3 2 2 3 3 3 4 1;
      3 1 4 4 4 4 4 4 2 4]

@test [counthits(gt, rs, k) for k=1:5] == [3, 8, 8, 10, 10]

@test counthits(gt, rs, 1:3) == [3, 8, 8]
@test counthits(gt, rs, [2, 4]) == [8, 10]
@test counthits(gt, rs, 1:2:5) == [3, 8, 10]

@test [hitrate(gt, rs, k) for k=1:5] ≈ [0.3, 0.8, 0.8, 1.0, 1.0]
@test hitrates(gt, rs, 1:3) ≈ [0.3, 0.8, 0.8]
@test hitrates(gt, rs, [2, 4]) ≈ [0.8, 1.0]
@test hitrates(gt, rs, 1:2:5) ≈ [0.3, 0.8, 1.0]

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
@test f1score(r) ≈ harmmean([recall(r), precision(r)])


## auxiliary: find_thresbin & lin_threshold

ts = [2, 4, 6, 8]
ft = [MLBase.find_thresbin(i, ts) for i = 1:9]
@test isa(ft, Vector{Int})
@test ft == [1, 2, 2, 3, 3, 4, 4, 5, 5]

@test MLBase.lin_thresholds([1, 5], 5, Forward) == 1.0:1.0:5.0
@test MLBase.lin_thresholds([1, 5], 5, Reverse) == 5.0:-1.0:1.0


## roc

gt = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
ss = [1., 2., 3., 4., 5., 4., 5., 6., 7., 8., 9.]

r2 = ROCNums{Int}(6, 5, 6, 1, 4, 0)
r4 = ROCNums{Int}(6, 5, 6, 3, 2, 0)
r6 = ROCNums{Int}(6, 5, 4, 5, 0, 2)

@test roc(gt, ss, 2.0) == r2
@test roc(gt, ss, 4.0) == r4
@test roc(gt, ss, 6.0) == r6
@test roc(gt, ss, 2.0, Forward) == r2
@test roc(gt, ss, 4.0, Forward) == r4
@test roc(gt, ss, 6.0, Forward) == r6

@test roc(gt, ss, [2.0, 4.0, 6.0], Forward) == [r2, r4, r6]
@test roc(gt, ss, [2.0, 4.0, 6.0]) == [r2, r4, r6]

@test roc(gt, ss, 5) == roc(gt, ss, [1., 3., 5., 7., 9.])
@test roc(gt, ss) == roc(gt, ss, 100)

gt = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
ss = [1., 2., 3., 4., 5., 4., 5., 6., 7., 8., 9.]

r2 = ROCNums{Int}(5, 6, 2, 6, 0, 3)
r4 = ROCNums{Int}(5, 6, 4, 5, 1, 1)
r6 = ROCNums{Int}(5, 6, 5, 3, 3, 0)

@test roc(gt, ss, 2.0, Reverse) == r2
@test roc(gt, ss, 4.0, Reverse) == r4
@test roc(gt, ss, 6.0, Reverse) == r6

@test roc(gt, ss, [6.0, 4.0, 2.0], Reverse) == [r6, r4, r2]

gt = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
pr = [1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1]
ss = [0.2, 0.2, 0.2, 0.3, 0.3, 0.6, 0.6, 0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8]

r00 = roc(gt, (pr, ss), 0.00)
r25 = roc(gt, (pr, ss), 0.25)
r50 = roc(gt, (pr, ss), 0.50)
r75 = roc(gt, (pr, ss), 0.75)
r100 = roc(gt, (pr, ss), 1.00)

@test r00 == ROCNums{Int}(10, 5, 8, 0, 5, 0)
@test r25 == ROCNums{Int}(10, 5, 8, 3, 2, 0)
@test r50 == ROCNums{Int}(10, 5, 8, 5, 0, 0)
@test r75 == ROCNums{Int}(10, 5, 4, 5, 0, 5)
@test r100 == ROCNums{Int}(10, 5, 0, 5, 0, 10)

@test roc(gt, (pr, ss), 0.0:0.25:1.0) == [r00, r25, r50, r75, r100]
# @test roc(gt, (pr, ss), 7) == roc(gt, (pr, ss), 0.2:0.1:0.8, Forward)
@test roc(gt, (pr, ss)) == roc(gt, (pr, ss), MLBase.lin_thresholds([0.2, 0.8], 100, Forward))
