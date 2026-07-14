module Precompilation

export precompile_indexmanipulations, precompile_contract, precompile_factorizations

using ..TensorKit
using ..TensorKit: TO
using VectorInterface: One, Zero
using TensorOperations: @tensor
using PrecompileTools: @setup_workload, @compile_workload
using Preferences: @load_preference

# Preferences
# -----------
function _validate_precompile_eltypes(eltypes)
    eltypes isa Vector{String} ||
        throw(ArgumentError("`precompile_eltypes` should be a vector of strings, got $(typeof(eltypes)) instead"))
    return map(eltypes) do Tstr
        T = eval(Meta.parse(Tstr))
        (T isa DataType && T <: Number) ||
            error("Invalid precompile_eltypes entry: `$Tstr`")
        return T
    end
end

const PRECOMPILE_ELTYPES = _validate_precompile_eltypes(
    @load_preference("precompile_eltypes", ["Float64", "ComplexF64"])
)

const PRECOMPILE_NDIMS = let n = @load_preference("precompile_ndims", 4)
    (n isa Int && n ≥ 1) ||
        throw(ArgumentError("`precompile_ndims` should be a positive `Int`, got `$n`"))
    n
end

function _precompile_space(name::AbstractString)
    I = Base.eval(TensorKit, Meta.parse(name))
    (I isa DataType && I <: Sector) ||
        error("invalid `precompile_sectors` entry `$name`; expected a `Sector` type")
    return oneunit(Vect[I])
end

const PRECOMPILE_SECTORS = let s = @load_preference(
        "precompile_sectors", map(x -> repr(x; context = :module => TensorKit), [Trivial, Z2Irrep, SU2Irrep, FermionParity])
    )
    s isa Vector{String} ||
        throw(ArgumentError("`precompile_sectors` should be a vector of strings, got $(typeof(s))"))
    s
end

const PRECOMPILE_CONTRACT = @load_preference("precompile_contract", true)
const PRECOMPILE_INDEXMANIPULATIONS = @load_preference("precompile_indexmanipulations", true)
const PRECOMPILE_FACTORIZATIONS = @load_preference("precompile_factorizations", true)

# Workload
# --------
include("precompile/indexmanipulations.jl")
include("precompile/contract.jl")
include("precompile/factorizations.jl")

@setup_workload begin
    for name in PRECOMPILE_SECTORS
        V = _precompile_space(name)
        @compile_workload begin
            PRECOMPILE_INDEXMANIPULATIONS && precompile_indexmanipulations(V; eltypes = PRECOMPILE_ELTYPES)
            PRECOMPILE_CONTRACT && precompile_contract(V; eltypes = PRECOMPILE_ELTYPES)
            PRECOMPILE_FACTORIZATIONS && precompile_factorizations(V; eltypes = PRECOMPILE_ELTYPES)
        end
    end
end

end # module Precompilation
