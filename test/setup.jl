module TestSetup

export randindextuple, randcircshift, _repartition, trivtuple
export default_tol
export smallset, randsector, hasfusiontensor, force_planar
export random_fusion
export sectorlist, fast_sectorlist
export test_dim_isapprox
export Vtr, Vℤ₂, Vfℤ₂, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂, VfSU₂, VSU₂U₁, Vfib, VIB_diag, VIB_M
export default_spacelist, factorization_spacelist
export remove_qrgauge_dependence!, remove_lqgauge_dependence!
export remove_eiggauge_dependence!, remove_eighgauge_dependence!, remove_svdgauge_dependence!
export test_ad_rrule, _project_hermitian
export _isunitary, _isone

using Random
using Test: @test
using TensorKit
using TensorKit: ℙ, PlanarTrivial
using TensorKitSectors
using TensorOperations: IndexTuple, Index2Tuple
using Base.Iterators: take, product
using TupleTools
using MatrixAlgebraKit: MatrixAlgebraKit, diagview
using ChainRulesCore: NoTangent
using ChainRulesTestUtils: ChainRulesTestUtils, test_rrule
using Zygote: Zygote, rrule_via_ad

Random.seed!(123456)

# IndexTuple utility
# ------------------
function randindextuple(N::Int, k::Int = rand(0:N))
    @assert 0 ≤ k ≤ N
    _p = randperm(N)
    return (tuple(_p[1:k]...), tuple(_p[(k + 1):end]...))
end
function randcircshift(N₁::Int, N₂::Int, k::Int = rand(0:(N₁ + N₂)))
    N = N₁ + N₂
    @assert 0 ≤ k ≤ N
    p = TupleTools.vcat(ntuple(identity, N₁), reverse(ntuple(identity, N₂) .+ N₁))
    n = rand(0:N)
    _p = TupleTools.circshift(p, n)
    return (tuple(_p[1:k]...), reverse(tuple(_p[(k + 1):end]...)))
end

trivtuple(N) = ntuple(identity, N)

Base.@constprop :aggressive function _repartition(p::IndexTuple, N₁::Int)
    length(p) >= N₁ ||
        throw(ArgumentError("cannot repartition $(typeof(p)) to $N₁, $(length(p) - N₁)"))
    return TupleTools.getindices(p, trivtuple(N₁)),
        TupleTools.getindices(p, trivtuple(length(p) - N₁) .+ N₁)
end
Base.@constprop :aggressive function _repartition(p::Index2Tuple, N₁::Int)
    return _repartition(linearize(p), N₁)
end
function _repartition(p::Union{IndexTuple, Index2Tuple}, ::Index2Tuple{N₁}) where {N₁}
    return _repartition(p, N₁)
end
function _repartition(p::Union{IndexTuple, Index2Tuple}, t::AbstractTensorMap)
    return _repartition(p, TensorKit.numout(t))
end

# Float32 and finite differences don't mix well
default_tol(::Type{<:Union{Float32, Complex{Float32}}}) = 1.0e-2
default_tol(::Type{<:Union{Float64, Complex{Float64}}}) = 1.0e-5

# Sector utility
# --------------
smallset(::Type{I}) where {I <: Sector} = take(values(I), 5)
function smallset(::Type{ProductSector{Tuple{I1, I2}}}) where {I1, I2}
    iter = product(smallset(I1), smallset(I2))
    s = collect(i ⊠ j for (i, j) in iter if dim(i) * dim(j) <= 6)
    return length(s) > 6 ? rand(s, 6) : s
end
function smallset(::Type{ProductSector{Tuple{I1, I2, I3}}}) where {I1, I2, I3}
    iter = product(smallset(I1), smallset(I2), smallset(I3))
    s = collect(i ⊠ j ⊠ k for (i, j, k) in iter if dim(i) * dim(j) * dim(k) <= 6)
    return length(s) > 6 ? rand(s, 6) : s
end
function randsector(::Type{I}) where {I <: Sector}
    s = collect(smallset(I))
    a = rand(s)
    while isunit(a) # don't use trivial label
        a = rand(s)
    end
    return a
end
function hasfusiontensor(I::Type{<:Sector})
    isa(UnitStyle(I), GenericUnit) && return false
    try
        TensorKit.fusiontensor(unit(I), unit(I), unit(I))
        return true
    catch e
        if e isa MethodError
            return false
        else
            rethrow(e)
        end
    end
end

"""
    force_planar(obj)

Replace an object with a planar equivalent -- i.e. one that disallows braiding.
"""
force_planar(V::ComplexSpace) = isdual(V) ? (ℙ^dim(V))' : ℙ^dim(V)
function force_planar(V::GradedSpace)
    return GradedSpace((c ⊠ PlanarTrivial() => dim(V, c) for c in sectors(V))..., isdual(V))
end
force_planar(V::ProductSpace) = mapreduce(force_planar, ⊗, V)
function force_planar(tsrc::TensorMap{<:Any, ComplexSpace})
    tdst = TensorMap{scalartype(tsrc)}(
        undef,
        force_planar(codomain(tsrc)) ←
            force_planar(domain(tsrc))
    )
    copyto!(block(tdst, PlanarTrivial()), block(tsrc, Trivial()))
    return tdst
end
function force_planar(tsrc::TensorMap{<:Any, <:GradedSpace})
    tdst = TensorMap{scalartype(tsrc)}(
        undef,
        force_planar(codomain(tsrc)) ←
            force_planar(domain(tsrc))
    )
    for (c, b) in blocks(tsrc)
        copyto!(block(tdst, c ⊠ PlanarTrivial()), b)
    end
    return tdst
end

function random_fusion(I::Type{<:Sector}, ::Val{N}) where {N} # for fusion tree tests
    N == 1 && return (randsector(I),)
    tail = random_fusion(I, Val(N - 1))
    s = randsector(I)
    counter = 0
    while isempty(⊗(s, first(tail))) && counter < 20
        counter += 1
        s = (counter < 20) ? randsector(I) : leftunit(first(tail))
    end
    return (s, tail...)
end

# helper function to check that d - dim(c) < dim(V) <= d where c is the largest sector
# to allow for truncations to have some margin with larger sectors
function test_dim_isapprox(V::ElementarySpace, d::Int)
    dim_c_max = maximum(dim, sectors(V); init = 1)
    return @test max(0, d - dim_c_max) ≤ dim(V) ≤ d + dim_c_max
end
function test_dim_isapprox(V::ProductSpace, d::Int)
    dim_c_max = maximum(dim, blocksectors(V); init = 1)
    return @test max(0, d - dim_c_max) ≤ dim(V) ≤ d + dim_c_max
end

_isunitary(x::Number; kwargs...) = isapprox(x * x', one(x); kwargs...)
_isunitary(x; kwargs...) = isunitary(x; kwargs...)
_isone(x; kwargs...) = isapprox(x, one(x); kwargs...)


uniquefusionsectorlist = (
    Z2Irrep, Z3Irrep, Z4Irrep, Z3Irrep ⊠ Z4Irrep,
    U1Irrep, FermionParity, FermionParity ⊠ FermionParity, FermionNumber,
    Z3Element{1}, ZNElement{5, 2},
)
simplefusionsectorlist = (
    CU1Irrep, SU2Irrep, FibonacciAnyon, IsingAnyon,
    FermionParity ⊠ U1Irrep ⊠ SU2Irrep, FermionParity ⊠ SU2Irrep ⊠ SU2Irrep,
    Z3Element{1} ⊠ FibonacciAnyon ⊠ FibonacciAnyon,
)
genericfusionsectorlist = (
    A4Irrep, A4Irrep ⊠ FermionParity, A4Irrep ⊠ SU2Irrep, A4Irrep ⊠ Z3Element{2}, A4Irrep ⊠ A4Irrep,
)
multifusionsectorlist = (
    IsingBimodule, IsingBimodule ⊠ SU2Irrep, IsingBimodule ⊠ IsingBimodule, IsingBimodule ⊠ Z3Element{1}, IsingBimodule ⊠ FibonacciAnyon ⊠ A4Irrep,
)

sectorlist = (
    uniquefusionsectorlist...,
    simplefusionsectorlist...,
    genericfusionsectorlist...,
    multifusionsectorlist...,
)
fast_sectorlist = (Z2Irrep, SU2Irrep, FermionParity ⊠ U1Irrep ⊠ SU2Irrep, FibonacciAnyon)

# spaces
Vtr = (ℂ^2, (ℂ^3)', ℂ^4, ℂ^3, (ℂ^2)')
Vℤ₂ = (
    Vect[Z2Irrep](0 => 2, 1 => 1),
    Vect[Z2Irrep](0 => 1, 1 => 2)',
    Vect[Z2Irrep](0 => 3, 1 => 2)',
    Vect[Z2Irrep](0 => 2, 1 => 3),
    Vect[Z2Irrep](0 => 2, 1 => 5),
)
Vfℤ₂ = (
    Vect[FermionParity](0 => 2, 1 => 1),
    Vect[FermionParity](0 => 1, 1 => 2)',
    Vect[FermionParity](0 => 2, 1 => 1)',
    Vect[FermionParity](0 => 2, 1 => 3),
    Vect[FermionParity](0 => 2, 1 => 5),
)
Vℤ₃ = (
    Vect[Z3Irrep](0 => 1, 1 => 2, 2 => 1),
    Vect[Z3Irrep](0 => 2, 1 => 1, 2 => 1),
    Vect[Z3Irrep](0 => 1, 1 => 2, 2 => 1)',
    Vect[Z3Irrep](0 => 1, 1 => 2, 2 => 3),
    Vect[Z3Irrep](0 => 1, 1 => 3, 2 => 3)',
)
VU₁ = (
    Vect[U1Irrep](0 => 1, 1 => 2, -1 => 2),
    Vect[U1Irrep](0 => 3, 1 => 1, -1 => 1),
    Vect[U1Irrep](0 => 2, 1 => 2, -1 => 1)',
    Vect[U1Irrep](0 => 1, 1 => 2, -1 => 3),
    Vect[U1Irrep](0 => 1, 1 => 3, -1 => 3)',
)
VfU₁ = (
    Vect[FermionNumber](0 => 1, 1 => 2, -1 => 2),
    Vect[FermionNumber](0 => 3, 1 => 1, -1 => 1),
    Vect[FermionNumber](0 => 2, 1 => 2, -1 => 1)',
    Vect[FermionNumber](0 => 1, 1 => 2, -1 => 3),
    Vect[FermionNumber](0 => 1, 1 => 3, -1 => 3)',
)
VCU₁ = (
    Vect[CU1Irrep]((0, 0) => 1, (0, 1) => 2, 1 => 1),
    Vect[CU1Irrep]((0, 0) => 3, (0, 1) => 0, 1 => 1),
    Vect[CU1Irrep]((0, 0) => 1, (0, 1) => 0, 1 => 2)',
    Vect[CU1Irrep]((0, 0) => 2, (0, 1) => 2, 1 => 1),
    Vect[CU1Irrep]((0, 0) => 2, (0, 1) => 1, 1 => 2)',
)
VSU₂ = (
    Vect[SU2Irrep](0 => 3, 1 // 2 => 1),
    Vect[SU2Irrep](0 => 2, 1 => 1),
    Vect[SU2Irrep](1 // 2 => 1, 1 => 1)',
    Vect[SU2Irrep](0 => 2, 1 // 2 => 2),
    Vect[SU2Irrep](0 => 1, 1 // 2 => 1, 3 // 2 => 1)',
)
VfSU₂ = (
    Vect[FermionSpin](0 => 3, 1 // 2 => 1),
    Vect[FermionSpin](0 => 2, 1 => 1),
    Vect[FermionSpin](1 // 2 => 1, 1 => 1)',
    Vect[FermionSpin](0 => 2, 1 // 2 => 2),
    Vect[FermionSpin](0 => 1, 1 // 2 => 1, 3 // 2 => 1)',
)
VSU₂U₁ = (
    Vect[SU2Irrep ⊠ U1Irrep]((0, 0) => 1, (1 // 2, -1) => 1),
    Vect[SU2Irrep ⊠ U1Irrep](
        (0, 0) => 2, (0, 2) => 1, (1, 0) => 1, (1, -2) => 1,
        (1 // 2, -1) => 1
    ),
    Vect[SU2Irrep ⊠ U1Irrep]((1 // 2, 1) => 1, (1, -2) => 1)',
    Vect[SU2Irrep ⊠ U1Irrep]((0, 0) => 2, (0, 2) => 1, (1 // 2, 1) => 1),
    Vect[SU2Irrep ⊠ U1Irrep]((0, 0) => 1, (1 // 2, 1) => 1)',
)
Vfib = (
    Vect[FibonacciAnyon](:I => 1, :τ => 1),
    Vect[FibonacciAnyon](:I => 1, :τ => 2)',
    Vect[FibonacciAnyon](:I => 3, :τ => 2)',
    Vect[FibonacciAnyon](:I => 2, :τ => 3),
    Vect[FibonacciAnyon](:I => 2, :τ => 2),
)

C0, C1 = IsingBimodule(1, 1, 0), IsingBimodule(1, 1, 1)
D0, D1 = IsingBimodule(2, 2, 0), IsingBimodule(2, 2, 1)
M, Mop = IsingBimodule(1, 2, 0), IsingBimodule(2, 1, 0)
VIB_diag = (
    Vect[IsingBimodule](C0 => 1, C1 => 2),
    Vect[IsingBimodule](C0 => 2, C1 => 1),
    Vect[IsingBimodule](C0 => 3, C1 => 1),
    Vect[IsingBimodule](C0 => 2, C1 => 3),
    Vect[IsingBimodule](C0 => 3, C1 => 2),
)

# not a random ordering! designed to make V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5 work (tensors)
# while V1 ⊗ V2 ← V4 isn't empty (factorizations)
VIB_M = (
    Vect[IsingBimodule](C0 => 1, C1 => 2),
    Vect[IsingBimodule](M => 3),
    Vect[IsingBimodule](C0 => 2, C1 => 3),
    Vect[IsingBimodule](M => 4),
    Vect[IsingBimodule](D0 => 3, D1 => 4),
)

# Spacelist selection
# -------------------
function default_spacelist(fast_tests::Bool)
    fast_tests && return (Vtr, Vℤ₃, VSU₂)
    if get(ENV, "CI", "false") == "true"
        println("Detected running on CI")
        Sys.iswindows() && return (Vtr, Vℤ₂, Vfℤ₂, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂, VIB_diag)
        Sys.isapple()   && return (Vtr, Vℤ₂, Vfℤ₂, Vℤ₃, VfU₁, VfSU₂, VSU₂U₁, VIB_M)
        return (Vtr, Vℤ₂, Vfℤ₂, VU₁, VCU₁, VSU₂, VfSU₂, VSU₂U₁, VIB_diag, VIB_M)
    end
    return (Vtr, Vℤ₂, Vfℤ₂, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂, VfSU₂, VSU₂U₁, VIB_diag, VIB_M)
end

function factorization_spacelist(fast_tests::Bool)
    fast_tests && return (Vtr, Vℤ₃, VSU₂)
    if get(ENV, "CI", "false") == "true"
        println("Detected running on CI")
        Sys.iswindows() && return (Vtr, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂, VIB_diag)
        Sys.isapple()   && return (Vtr, Vℤ₃, VfU₁, VfSU₂, VIB_M)
        return (Vtr, VU₁, VCU₁, VSU₂, VfSU₂, VIB_diag, VIB_M)
    end
    return (Vtr, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂, VfSU₂, VIB_diag, VIB_M)
end

# Gauge-fixing tangents for AD factorization tests
# -------------------------------------------------
function remove_qrgauge_dependence!(ΔQ, t, Q)
    for (c, b) in blocks(ΔQ)
        m, n = size(block(t, c))
        minmn = min(m, n)
        Qc = block(Q, c)
        Q1 = view(Qc, 1:m, 1:minmn)
        ΔQ2 = view(b, :, (minmn + 1):m)
        mul!(ΔQ2, Q1, Q1' * ΔQ2)
    end
    return ΔQ
end
function remove_lqgauge_dependence!(ΔQ, t, Q)
    for (c, b) in blocks(ΔQ)
        m, n = size(block(t, c))
        minmn = min(m, n)
        Qc = block(Q, c)
        Q1 = view(Qc, 1:minmn, 1:n)
        ΔQ2 = view(b, (minmn + 1):n, :)
        mul!(ΔQ2, ΔQ2 * Q1', Q1)
    end
    return ΔQ
end
function remove_eiggauge_dependence!(
        ΔV, D, V; degeneracy_atol = MatrixAlgebraKit.default_pullback_degeneracy_atol(D)
    )
    gaugepart = V' * ΔV
    for (c, b) in blocks(gaugepart)
        Dc = diagview(block(D, c))
        # for some reason this fails only on tests, and I cannot reproduce it in an
        # interactive session.
        # b[abs.(transpose(diagview(Dc)) .- diagview(Dc)) .>= degeneracy_atol] .= 0
        for j in axes(b, 2), i in axes(b, 1)
            abs(Dc[i] - Dc[j]) >= degeneracy_atol && (b[i, j] = 0)
        end
    end
    mul!(ΔV, V / (V' * V), gaugepart, -1, 1)
    return ΔV
end
function remove_eighgauge_dependence!(
        ΔV, D, V; degeneracy_atol = MatrixAlgebraKit.default_pullback_degeneracy_atol(D)
    )
    gaugepart = project_antihermitian!(V' * ΔV)
    for (c, b) in blocks(gaugepart)
        Dc = diagview(block(D, c))
        # for some reason this fails only on tests, and I cannot reproduce it in an
        # interactive session.
        # b[abs.(transpose(diagview(Dc)) .- diagview(Dc)) .>= degeneracy_atol] .= 0
        for j in axes(b, 2), i in axes(b, 1)
            abs(Dc[i] - Dc[j]) >= degeneracy_atol && (b[i, j] = 0)
        end
    end
    mul!(ΔV, V, gaugepart, -1, 1)
    return ΔV
end
function remove_svdgauge_dependence!(
        ΔU, ΔVᴴ, U, S, Vᴴ; degeneracy_atol = MatrixAlgebraKit.default_pullback_degeneracy_atol(S)
    )
    gaugepart = project_antihermitian!(U' * ΔU + Vᴴ * ΔVᴴ')
    for (c, b) in blocks(gaugepart)
        Sd = diagview(block(S, c))
        # for some reason this fails only on tests, and I cannot reproduce it in an
        # interactive session.
        # b[abs.(transpose(diagview(Sc)) .- diagview(Sc)) .>= degeneracy_atol] .= 0
        for j in axes(b, 2), i in axes(b, 1)
            abs(Sd[i] - Sd[j]) >= degeneracy_atol && (b[i, j] = 0)
        end
    end
    mul!(ΔU, U, gaugepart, -1, 1)
    return ΔU, ΔVᴴ
end

# ChainRules test utilities
# -------------------------
function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::AbstractTensorMap)
    return randn!(similar(x))
end
function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::DiagonalTensorMap)
    V = x.domain
    return DiagonalTensorMap(randn(eltype(x), reduceddim(V)), V)
end
ChainRulesTestUtils.rand_tangent(::AbstractRNG, ::VectorSpace) = NoTangent()
function ChainRulesTestUtils.test_approx(
        actual::AbstractTensorMap, expected::AbstractTensorMap, msg = ""; kwargs...
    )
    for (c, b) in blocks(actual)
        ChainRulesTestUtils.@test_msg msg isapprox(b, block(expected, c); kwargs...)
    end
    return nothing
end

function test_ad_rrule(f, args...; check_inferred = false, kwargs...)
    test_rrule(
        Zygote.ZygoteRuleConfig(), f, args...;
        rrule_f = rrule_via_ad, check_inferred, kwargs...
    )
    return nothing
end

# project_hermitian is non-differentiable for now
_project_hermitian(x) = (x + x') / 2

end
