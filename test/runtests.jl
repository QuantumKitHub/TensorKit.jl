# ARGS parsing
# ------------
using ArgParse: ArgParse
using SafeTestsets: @safetestset

function parse_commandline(args = ARGS)
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table! s begin
        "--groups"
        action => :store_arg
        nargs => '+'
        arg_type = String
    end
    return ArgParse.parse_args(args, s; as_symbols = true)
end

settings = parse_commandline()

if isempty(settings[:groups])
    if haskey(ENV, "GROUP")
        groups = [ENV["GROUP"]]
    else
        groups = filter(isdir ∘ Base.Fix1(joinpath, @__DIR__), readdir(@__DIR__))
    end
else
    groups = settings[:groups]
end

checktestgroup(group) = isdir(joinpath(@__DIR__, group)) ||
    throw(ArgumentError("Invalid group ($group), no such folder"))
foreach(checktestgroup, groups)

@info "Loaded test groups:" groups

# don't run all tests on GPU, only the GPU specific ones
is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

# Run test groups
# ---------------

"match files of the form `*.jl`, but exclude `*setup*.jl`"
istestfile(fn) = endswith(fn, ".jl") && !contains(fn, "setup")

# process test groups
@time for group in groups
    @info "Running test group: $group"

    # handle GPU cases separately
    if group == "cuda"
        using CUDA
        CUDA.functional() || continue
        @time include("cuda/tensors.jl")
        @time include("cuda/factorizations.jl")
    elseif group == "amd"
        using AMDGPU
        AMDGPU.functional() || continue
        #@time include("amd/tensors.jl")
        @time include("amd/factorizations.jl")
    elseif is_buildkite
        continue
    end

    # somehow AD tests are unreasonably slow on Apple CI
    # and ChainRulesTestUtils doesn't like prereleases
    if group == "autodiff"
        Sys.isapple() && get(ENV, "CI", "false") == "true" && continue
        isempty(VERSION.prerelease) || continue
    end

    grouppath = joinpath(@__DIR__, group)
    @time for file in filter(istestfile, readdir(grouppath))
        @info "Running test file: $file"
        filepath = joinpath(grouppath, file)
        @eval @safetestset $file begin
            include($filepath)
        end
    end
end
