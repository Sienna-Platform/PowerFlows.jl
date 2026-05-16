# PowerFlows.jl

[![Main - CI](https://github.com/Sienna-Platform/PowerFlows.jl/actions/workflows/main-tests.yml/badge.svg)](https://github.com/Sienna-Platform/PowerFlows.jl/actions/workflows/main-tests.yml)
[![codecov](https://codecov.io/gh/Sienna-Platform/PowerFlows.jl/branch/main/graph/badge.svg?token=W2GIoyXQcF)](https://codecov.io/gh/Sienna-Platform/PowerFlows.jl)
[![Documentation](https://github.com/Sienna-Platform/PowerFlows.jl/actions/workflows/docs.yml/badge.svg)](https://sienna-platform.github.io/PowerFlows.jl/dev/)
[<img src="https://img.shields.io/badge/slack-@Sienna/PF-sienna.svg?logo=slack">](https://join.slack.com/t/core-sienna/shared_invite/zt-glam9vdu-o8A9TwZTZqqNTKHa7q3BpQ)
[![PowerFlows Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FPowerFlows&query=total_requests&label=Downloads)](http://juliapkgstats.com/pkg/PowerFlows)

This package is under development at the moment. The current version is a port of the existing Power Flow solve in `PowerSystems.jl`.

The objective of this package is to provide an interface to multiple solution methods for the Power Flow problem along with utilities common in commercial software like Siemens PSS/e and GE PSLF.

During development the API will change

## Development

Contributions to the development and enhancement of PowerFlows is welcome. Please see [CONTRIBUTING.md](https://github.com/Sienna-Platform/PowerFlows.jl/blob/master/CONTRIBUTING.md) for code contribution guidelines.

## License

PowerFlows is released under a BSD [license](https://github.com/NREL/PowerFlows.jl/blob/master/LICENSE). PowerFlows has been developed as part of the Scalable Integrated Infrastructure Planning (SIIP)
initiative at the U.S. Department of Energy's National Laboratory of the Rockies ([NLR](https://www.nlr.gov/), formerly NREL)
