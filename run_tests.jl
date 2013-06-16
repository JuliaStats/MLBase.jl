
tests = ["vecarith", "vecreduc", "norms", "intstats", "pdmat"]

for t in tests
	fp = joinpath("test", "test_$t.jl")
	println("running $fp ...")
	include(fp)
end

