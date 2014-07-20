## Change Logs

#### From v0.3 to v0.4

- Move deviation computing functions to [StatsBase](https://github.com/JuliaStats/StatsBase.jl)
- Move ``countne`` and ``counteq`` to [StatsBase](https://github.com/JuliaStats/StatsBase.jl)
- Deprecate ``Standardize`` (in favor of [StatsBase](https://github.com/JuliaStats/StatsBase.jl)'s ``zscore``)

#### From v0.4 to v0.5

- Move documentation from Readme to [Sphinx Docs](http://mlbasejl.readthedocs.org/en/latest/)
- ``cross_validate`` now returns a vector of scores (see [here](http://mlbasejl.readthedocs.org/en/latest/crossval.html#cross_validate)).
