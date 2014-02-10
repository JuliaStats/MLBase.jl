
using MLBase
using Base.Test

## Kfold

ss = collect(Kfold(12, 3))
@test length(ss) == 3
for i = 1:3
	@test isa(ss[i], Vector{Int})
	@test issorted(ss[i])
end
x = vcat(ss...)
@test sort(x) == [1:12]

## LOOCV

ss = collect(LOOCV(4))
@test length(ss) == 4

@test ss[1] == [2, 3, 4]
@test ss[2] == [1, 3, 4]
@test ss[3] == [1, 2, 4]
@test ss[4] == [1, 2, 3]

## RandomSub

# temporary disable until StatsBase passes travis

# ss = collect(RandomSub(10, 5, 6))
# @test length(ss) == 6
# for i = 1:6
# 	@test length(ss[i]) == 5
# 	@test 1 <= minimum(ss[i]) <= maximum(ss[i]) <= 10
# 	@test length(unique(ss[i])) == 5
# end

