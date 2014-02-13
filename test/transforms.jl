
using MLBase
using Base.Test

# standardize

X = rand(5, 8)

(Y, t) = standardize(X; center=false, scale=false)
@test isa(t, Standardize)
@test isempty(t.mean)
@test isempty(t.scale)
@test isequal(X, Y)
@test_approx_eq transform(t, X[:,1]) Y[:,1]

(Y, t) = standardize(X; center=false, scale=true)
@test isa(t, Standardize)
@test isempty(t.mean)
@test length(t.scale) == 5
s = sqrt(sum(abs2(X), 2) ./ (8 - 1))
@test_approx_eq Y X ./ s
@test_approx_eq transform(t, X[:,1]) Y[:,1]

(Y, t) = standardize(X; center=true, scale=false)
@test isa(t, Standardize)
@test length(t.mean) == 5
@test isempty(t.scale)
@test_approx_eq Y X .- mean(X, 2)
@test_approx_eq transform(t, X[:,1]) Y[:,1]

(Y, t) = standardize(X; center=true, scale=true)
@test isa(t, Standardize)
@test length(t.mean) == 5
@test length(t.scale) == 5
@test_approx_eq Y (X .- mean(X, 2)) ./ std(X, 2)
@test_approx_eq transform(t, X[:,1]) Y[:,1]
