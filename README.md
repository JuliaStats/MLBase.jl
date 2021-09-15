# MLBase.jl

Swiss knife for machine learning.

[![Build Status](https://github.com/JuliaStats/MLBase.jl/workflows/CI/badge.svg)](https://github.com/JuliaStats/MLBase.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![Coveralls](https://coveralls.io/repos/github/JuliaStats/MLBase.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaStats/MLBase.jl?branch=master)
[![Documentation Status](https://readthedocs.org/projects/mlbasejl/badge/?version=latest)](http://mlbasejl.readthedocs.io/en/latest/?badge=latest)

This package does not implement specific machine learning algorithms. Instead, it provides a collection of useful tools to support machine learning programs, including:

- Data manipulation & preprocessing
- Score-based classification
- Performance evaluation (*e.g.* evaluating ROC)
- Cross validation
- Model tuning (*i.e.* search best settings of parameters)

**Notes:** This package depends on [StatsBase](https://github.com/JuliaStats/StatsBase.jl) and reexports all names therefrom.

### Resources

- **Documentation:** <http://mlbasejl.readthedocs.org/en/latest/>
- **Release Notes:** <https://github.com/JuliaStats/MLBase.jl/blob/master/NEWS.md>
