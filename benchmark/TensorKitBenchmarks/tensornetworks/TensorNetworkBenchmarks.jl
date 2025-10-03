module TensorNetworkBenchmarks

include(joinpath(@__DIR__, "..", "utils", "BenchUtils.jl"))

using .BenchUtils
using BenchmarkTools
using TensorKit
using TOML

const SUITE = BenchmarkGroup()
const all_parameters = TOML.parsefile(joinpath(@__DIR__, "benchparams.toml"))

# mpo contraction
# ---------------
function init_mpo_tensors(T, (Vmps, Vmpo, Vphys))
    A = Tensor(randn, T, Vmps ⊗ Vphys ⊗ Vmps')
    M = Tensor(randn, T, Vmpo ⊗ Vphys ⊗ Vphys' ⊗ Vmpo')
    FL = Tensor(randn, T, Vmps ⊗ Vmpo' ⊗ Vmps')
    FR = Tensor(randn, T, Vmps ⊗ Vmpo ⊗ Vmps')
    return A, M, FL, FR
end

function benchmark_mpo(A, M, FL, FR)
    return @tensor FL[4, 2, 1] * A[1, 3, 6] * M[2, 5, 3, 7] * conj(A[4, 5, 8]) * FR[6, 7, 8]
end

function benchmark_mpo!(benchgroup, params::Dict)
    haskey(benchgroup, "mpo") || addgroup!(benchgroup, "mpo")
    bench = benchgroup["mpo"]
    for kwargs in expand_kwargs(params)
        benchmark_mpo!(bench; kwargs...)
    end
    return nothing
end
function benchmark_mpo!(bench; sigmas = nothing, T = "Float64", I = "Trivial", dims)
    T_ = parse_type(T)
    I_ = parse_type(I)

    Vs = generate_space.(I_, dims, sigmas)
    init() = init_mpo_tensors(T_, Vs)

    bench[T, I, dims, sigmas] = @benchmarkable benchmark_mpo(A, M, FL, FR) setup = ((A, M, FL, FR) = $init())

    return nothing
end

if haskey(all_parameters, "mpo")
    for params in all_parameters["mpo"]
        benchmark_mpo!(SUITE, params)
    end
end

# pepo contraction
# ----------------
function init_pepo_tensors(T, (Vpeps, Vpepo, Vphys, Venv))
    A = Tensor(randn, T, Vpeps ⊗ Vpeps ⊗ Vphys ⊗ Vpeps' ⊗ Vpeps')
    P = Tensor(randn, T, Vpepo ⊗ Vpepo ⊗ Vphys ⊗ Vphys' ⊗ Vpepo' ⊗ Vpepo')
    FL = Tensor(randn, T, Venv ⊗ Vpeps ⊗ Vpepo' ⊗ Vpeps' ⊗ Venv')
    FD = Tensor(randn, T, Venv ⊗ Vpeps ⊗ Vpepo' ⊗ Vpeps' ⊗ Venv')
    FR = Tensor(randn, T, Venv ⊗ Vpeps ⊗ Vpepo ⊗ Vpeps' ⊗ Venv')
    FU = Tensor(randn, T, Venv ⊗ Vpeps ⊗ Vpepo ⊗ Vpeps' ⊗ Venv')
    return A, P, FL, FD, FR, FU
end

function benchmark_pepo(A, P, FL, FD, FR, FU)
    return @tensor FL[18, 7, 4, 2, 1] * FU[1, 3, 6, 9, 10] * A[2, 17, 5, 3, 11] *
        P[4, 16, 8, 5, 6, 12] * conj(A[7, 15, 8, 9, 13]) *
        FR[10, 11, 12, 13, 14] * FD[14, 15, 16, 17, 18]
end

function benchmark_pepo!(benchgroup, params::Dict)
    haskey(benchgroup, "pepo") || addgroup!(benchgroup, "pepo")
    bench = benchgroup["pepo"]
    for kwargs in expand_kwargs(params)
        benchmark_pepo!(bench; kwargs...)
    end
    return nothing
end
function benchmark_pepo!(bench; sigmas = nothing, T = "Float64", I = "Trivial", dims)
    T_ = parse_type(T)
    I_ = parse_type(I)

    Vs = generate_space.(I_, dims, sigmas)
    init() = init_pepo_tensors(T_, Vs)

    bench[T, I, dims, sigmas] = @benchmarkable benchmark_pepo(A, P, FL, FD, FR, FU) setup = ((A, P, FL, FD, FR, FU) = $init())

    return nothing
end

if haskey(all_parameters, "pepo")
    for params in all_parameters["pepo"]
        benchmark_pepo!(SUITE, params)
    end
end

# mera contraction
# ----------------
function init_mera_tensors(T, V)
    u = Tensor(randn, T, V ⊗ V ⊗ V' ⊗ V')
    w = Tensor(randn, T, V ⊗ V ⊗ V')
    ρ = Tensor(randn, T, V ⊗ V ⊗ V ⊗ V' ⊗ V' ⊗ V')
    h = Tensor(randn, T, V ⊗ V ⊗ V ⊗ V' ⊗ V' ⊗ V')
    return u, w, ρ, h
end

function benchmark_mera(u, w, ρ, h)
    return @tensor (
        (
            (
                (
                    (
                        ((h[9, 3, 4, 5, 1, 2] * u[1, 2, 7, 12]) * conj(u[3, 4, 11, 13])) *
                            (u[8, 5, 15, 6] * w[6, 7, 19])
                    ) * (conj(u[8, 9, 17, 10]) * conj(w[10, 11, 22]))
                ) * ((w[12, 14, 20] * conj(w[13, 14, 23])) * ρ[18, 19, 20, 21, 22, 23])
            ) * w[16, 15, 18]
        ) * conj(w[16, 17, 21])
    )
end

function benchmark_mera!(benchgroup, params::Dict)
    haskey(benchgroup, "mera") || addgroup!(benchgroup, "mera")
    bench = benchgroup["mera"]
    for kwargs in expand_kwargs(params)
        benchmark_mera!(bench; kwargs...)
    end
    return nothing
end

function benchmark_mera!(bench; sigmas = nothing, T = "Float64", I = "Trivial", dims)
    T_ = parse_type(T)
    I_ = parse_type(I)

    Vs = generate_space.(I_, dims, sigmas)
    init() = init_mera_tensors(T_, Vs)

    bench[T, I, dims, sigmas] = @benchmarkable benchmark_mera(u, w, ρ, h) setup = ((u, w, ρ, h) = $init())

    return nothing
end

if haskey(all_parameters, "mera")
    for params in all_parameters["mera"]
        benchmark_mera!(SUITE, params)
    end
end

end
