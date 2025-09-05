# TensorKit Benchmarks

This directory contains a benchmark suite for TensorKit.
Most of the benchmarks are designed to capture performance characteristics of the library, and are not intended to be used as a benchmark suite for comparing different libraries.
In particular, the main goal here is to catch performance regressions and/or improvements between different versions of TensorKit.

## Running the benchmarks

The benchmarks are written using `BenchmarkTools.jl`, and the full suite can be found in the `SUITE` global variable defined in `benchmarks.jl`.
Sometimes, it is useful to run only a subset of the benchmarks.
To do this, you can use the `--modules` flag to specify which modules to run.
Alternatively, you can use the `TensorKitBenchmarks` module directly, which is designed after `BaseBenchmarks` to allow for conditional loading of the benchmarks.

For a more streamlined CLI experience, you can use [`AirspeedVelocity.jl`](https://github.com/MilesCranmer/AirspeedVelocity.jl) to run the benchmarks.
The following command will run the benchmarks and compare with the current main branch:

```bash
benchpkg TensorKit \
    --rev=dirty,main \
    -o benchmark/results/ \
    -exeflags="--threads=4"
```

To compare with previous results, the following command can be used:

```bash
benchpkgtable TensorKit \
    --rev=dirty,main \
    -i benchmark/results/ \
    -o benchmark/results/ \
```
