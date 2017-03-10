# Tests of intstats.jl

using MLBase
using Base.Test

# classify

ss = rand(8, 50)
for i = 1:size(ss,2)
    ss_i = ss[:,i]
    kmax = indmax(ss_i)
    kmin = indmin(ss_i)
    vmax = ss_i[kmax]
    vmin = ss_i[kmin]

    @test classify(ss_i) == kmax
    @test classify(ss_i, Forward) == kmax
    @test classify(ss_i, Reverse) == kmin

    @test classify(ss_i, 0.8) == (vmax >= 0.8 ? kmax : 0)
    @test classify(ss_i, 0.8, Forward) == (vmax >= 0.8 ? kmax : 0)
    @test classify(ss_i, 0.2, Reverse) == (vmin <= 0.2 ? kmin : 0)

    @test classify_withscore(ss_i) == (kmax, ss_i[kmax])
    @test classify_withscore(ss_i, Forward) == (kmax, ss_i[kmax])
    @test classify_withscore(ss_i, Reverse) == (kmin, ss_i[kmin])
end

rmax = Int[indmax(ss[:,i]) for i = 1:size(ss,2)]
rmin = Int[indmin(ss[:,i]) for i = 1:size(ss,2)]
vmax = ss[sub2ind(size(ss), rmax, 1:size(ss,2))]
vmin = ss[sub2ind(size(ss), rmin, 1:size(ss,2))]

trmax = copy(rmax); trmax[vmax .< 0.8] = 0
trmin = copy(rmin); trmin[vmin .> 0.2] = 0

@test classify(ss) == rmax
@test classify(ss, Forward) == rmax
@test classify(ss, Reverse) == rmin

@test classify_withscores(ss) == (rmax, vmax)
@test classify_withscores(ss, Forward) == (rmax, vmax)
@test classify_withscores(ss, Reverse) == (rmin, vmin)

@test classify(ss, 0.8) == trmax
@test classify(ss, 0.8, Forward) == trmax
@test classify(ss, 0.2, Reverse) == trmin

# labelmap & labelencode

xs = ["a", "a", "b", "b", "a", "b", "c", "a"]
labels = [1, 1, 2, 2, 1, 2, 3, 1]
lmap = labelmap(xs)

@test keys(lmap) == ["a", "b", "c"]
@test labelencode(lmap, xs) == labels
@test labeldecode(lmap, labels) == xs

# groupindices

gs = Any[[1,2,5,8],[3,4,6],[7]]

@test groupindices(3, labels) == gs
@test groupindices(lmap, xs) == gs

# class encodings

@test BinaryClassEncoding <: ClassEncoding
@test ZeroOneClassEncoding <: BinaryClassEncoding
@test SignedClassEncoding <: BinaryClassEncoding

wrongDim1 = ["y","y","y"]
wrongDim2 = ["y","n","c"]

@test_throws ArgumentError ZeroOneClassEncoding(wrongDim1)
@test_throws ArgumentError ZeroOneClassEncoding(wrongDim2)
@test_throws ArgumentError SignedClassEncoding(wrongDim1)
@test_throws ArgumentError SignedClassEncoding(wrongDim2)

t = ["y", "n", "n", "y", "n"]
wrongLabel1 = ["a","b","a"]
wrongLabel2 = [1,2,2]

ce = ZeroOneClassEncoding(t)
@test_throws KeyError labelencode(ce, wrongLabel1)
@test_throws MethodError labelencode(ce, wrongLabel2)

ce = SignedClassEncoding(t)
@test_throws KeyError labelencode(ce, wrongLabel1)
@test_throws MethodError labelencode(ce, wrongLabel2)


t = ["y", "n", "n", "y", "n"]
y = ["n","y", "y", "n"]

ce = ZeroOneClassEncoding(t)
pred = labelencode(ce, y)
idx = groupindices(ce, t)
@test idx[1] == [1,4]
@test idx[2] == [2,3,5]
@test classDistribution(ce, t) == (["y","n"], [2, 3])
@test pred == [1, 0, 0, 1]
@test labeldecode(ce, pred) == y
@test labeldecode(ce, convert(Vector{Int},pred)) == y

ce = SignedClassEncoding(t)
pred = labelencode(ce, y)
idx = groupindices(ce, t)
@test idx[1] == [1,4]
@test idx[2] == [2,3,5]
@test classDistribution(ce, t) == (["y","n"], [2, 3])
@test pred == [1, -1, -1, 1]
@test labeldecode(ce, pred) == y
@test labeldecode(ce, convert(Vector{Int},pred)) == y


@test MultinomialClassEncoding <: ClassEncoding
@test MultivalueClassEncoding <: MultinomialClassEncoding
@test OneOfKClassEncoding <: MultinomialClassEncoding
@test OneHotClassEncoding <: MultinomialClassEncoding
@test OneHotClassEncoding == OneOfKClassEncoding

wrongDim1 = ["y","y","y"]

@test_throws ArgumentError MultivalueClassEncoding(wrongDim1)
@test_throws ArgumentError OneOfKClassEncoding(wrongDim1)

t = ["y", "n", "k", "y", "n"]
wrongLabel1 = ["a","b","a"]
wrongLabel2 = [1,2,2]

ce = MultivalueClassEncoding(t)
@test_throws KeyError labelencode(ce, wrongLabel1)
@test_throws MethodError labelencode(ce, wrongLabel2)

ce = OneOfKClassEncoding(t)
@test_throws KeyError labelencode(ce, wrongLabel1)
@test_throws MethodError labelencode(ce, wrongLabel2)

t = ["y", "n", "k", "y", "n"]
y = ["n", "y", "y", "k", "n"]

ce = MultivalueClassEncoding(t)
pred = labelencode(ce, y)
idx = groupindices(ce, t)
@test idx[1] == [1,4]
@test idx[2] == [2,5]
@test idx[3] == [3]
@test classDistribution(ce, t) == (["y","n","k"], [2, 2, 1])
@test pred == [2, 1, 1, 3, 2]
@test labeldecode(ce, pred) == y

ce = MultivalueClassEncoding(t, zero_based=true)
pred = labelencode(ce, y)
idx = groupindices(ce, t)
@test idx[1] == [1,4]
@test idx[2] == [2,5]
@test idx[3] == [3]
@test classDistribution(ce, t) == (["y","n","k"], [2, 2, 1])
@test pred == [1, 0, 0, 2, 1]
@test labeldecode(ce, pred) == y

ce = OneOfKClassEncoding(t)
pred = labelencode(ce, y)
@test pred ==
  [0  1  0;
   1  0  0;
   1  0  0;
   0  0  1;
   0  1  0]
@test labeldecode(ce, pred) == y
