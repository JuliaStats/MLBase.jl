
tests = ["vecarith", "vecreduc", "norms"]

for t in tests
	fp = joinpath("test", "test_$t.jl")
	println("running $fp ...")
	include(fp)
end

