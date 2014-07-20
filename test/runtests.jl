
tests = ["utils", 
         "classification", 
         "perfeval", 
         "crossval"]

for t in tests
    fp = "$t.jl"
    println("* running $fp ...")
    include(fp)
end

