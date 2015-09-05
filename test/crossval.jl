using MLBase
using Base.Test

## Kfold

ss = collect(Kfold(12, 3))
@test length(ss) == 3
for i = 1:3
    @test isa(ss[i], Vector{Int})
    @test issorted(ss[i])
end
x = vcat(map(s -> setdiff(1:12, s), ss)...)
@test sort(x) == collect(1:12)

## StratifiedKfold

strat = [:a, :a, :b, :b, :c, :c, :b, :c, :a]
@test_throws ErrorException StratifiedKfold(strat, 4)
ss = collect(StratifiedKfold(strat, 3))
for i in 1:2
    @test isa(ss[i], Vector{Int})
    @test issorted(ss[i])
    @test length(unique(strat[ss[i]])) == 3
end
x = vcat(map(s -> setdiff(1:9, s), ss)...)
@test sort(x) == collect(1:9)

## LOOCV

ss = collect(LOOCV(4))
@test length(ss) == 4

@test ss[1] == [2, 3, 4]
@test ss[2] == [1, 3, 4]
@test ss[3] == [1, 2, 4]
@test ss[4] == [1, 2, 3]

## Test LOOCV and Kfold are the same for n=k
@test Set(LOOCV(4)) == Set(Kfold(4,4))


## RandomSub

ss = collect(RandomSub(10, 5, 6))
@test length(ss) == 6
for i = 1:6
    @test length(ss[i]) == 5
    @test 1 <= minimum(ss[i]) <= maximum(ss[i]) <= 10
    @test length(unique(ss[i])) == 5
end

## StratifiedRandomSub

strat =  [:a, :a, :b, :b, :c, :c, :b, :c, :a]
@test_throws ErrorException StratifiedRandomSub(strat, 2, 10)
ss = collect(StratifiedRandomSub(strat, 6, 10))
@test length(ss) == 10
@test all(map(length, ss) .== 6)

#make sure small strata are (probably) always represented
strat = push!(repeach([:a], 100), :b)
ss = collect(StratifiedRandomSub(strat, 4, 100))
@test all(s -> :b ∈ strat[s], ss)


## cross validation

x0 = [1, 2, 3, 4]
scores = cross_validate(
            inds -> sum(inds),
            (m, inds) -> m / sum(inds),
            4, LOOCV(4))
@test scores == [9 / 1, 8 / 2, 7 / 3, 6 / 4]
