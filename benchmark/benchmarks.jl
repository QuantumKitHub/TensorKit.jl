# Load benchmark code
include("TensorKitBenchmarks/TensorKitBenchmarks.jl")
const SUITE = TensorKitBenchmarks.SUITE

# Populate benchmarks
# Detect if user supplied extra arguments to load only specific modules
# e.g. julia benchmarks.jl --modules=linalg,tensornetworks
modules_pattern = r"(?:--modules=)([\w,]+)"
arg_id = findfirst(contains(modules_pattern), ARGS)
if isnothing(arg_id)
    TensorKitBenchmarks.loadall!()
else
    modules = split(match(modules_pattern, ARGS[arg_id]).captures[1], ",")
    for m in modules
        TensorKitBenchmarks.load!(m)
    end
end
