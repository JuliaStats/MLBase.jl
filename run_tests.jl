
tests = ["vecarith", "vecreduc", "norms", "pdmat"]

for t in tests
	fp = joinpath("test", "test_$t.jl")
	println("running $fp ...")
	include(fp)
end

