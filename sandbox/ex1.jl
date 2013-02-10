
require("Options")
using OptionsMod
using PRML

o1 = @options max_iter=150
ko1 = kmeans_opts(o1)

println(ko1)

