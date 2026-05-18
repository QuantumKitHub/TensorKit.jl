# Based on the design of GPUArrays.jl

"""
    TestSuite

Suite of tests that may be used for all packages inheriting from TensorKit.

"""
module TestSuite

export zero_f, rand_f, randn_f

using Test
using TensorKit, TestExtras
using LinearAlgebra, Random, StableRNGs
using Adapt, AMDGPU, CUDA

const tests = Dict()

macro testsuite(name, ex)
    safe_name = lowercase(replace(replace(name, " " => "_"), "/" => "_"))
    fn = Symbol("test_", safe_name)
    return quote
        $(esc(fn))(AT; eltypes = supported_eltypes(AT, $(esc(fn)))) = $(esc(ex))(AT, eltypes)
        @assert !haskey(tests, $name) "testsuite already exists"
        tests[$name] = $fn
    end
end

testargs_summary(args...) = string(args)

const rng = StableRNG(123)
seed_rng!(seed) = Random.seed!(rng, seed)

zero_f(::Type{<:Vector}) = zeros
zero_f(::Type{<:CuVector}) = CUDA.zeros
zero_f(::Type{<:ROCVector}) = AMDGPU.zeros

rand_f(::Type{<:Vector}) = rand
rand_f(::Type{<:CuVector}) = cuRAND.rand
rand_f(::Type{<:ROCVector}) = AMDGPU.rand

randn_f(::Type{<:Vector}) = randn
randn_f(::Type{<:CuVector}) = cuRAND.randn
randn_f(::Type{<:ROCVector}) = AMDGPU.randn

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)

function hasfusiontensor(I::Type{<:Sector})
    try
        u = first(allunits(I))
        fusiontensor(u, u, u)
        return true
    catch e
        if e isa MethodError
            return false
        else
            rethrow(e)
        end
    end
end

include("tensors/construction.jl")
include("tensors/linalg.jl")

end
