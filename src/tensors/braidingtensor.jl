# BraidingTensor:
# special (2,2) tensor that implements a standard braiding operation
#====================================================================#
"""
    struct BraidingTensor{T, S <: IndexSpace, A <: DenseVector{T}} <: AbstractTensorMap{T, S, 2, 2}
    BraidingTensor(V1::S, V2::S, adjoint::Bool=false) where {S<:IndexSpace}
    BraidingTensor{T, S, A}(V1::S, V2::S, adjoint::Bool=false) where {T, S, A}

Specific subtype of [`AbstractTensorMap`](@ref) for representing the braiding tensor that
braids the first input over the second input; its inverse can be obtained as the adjoint.

It holds that `domain(BraidingTensor(V1, V2)) == V1 ⊗ V2` and
`codomain(BraidingTensor(V1, V2)) == V2 ⊗ V1`. The storage type `TA`
controls the array type of the braiding tensor used when indexing
and multiplying with other tensors.
"""
struct BraidingTensor{T, S, A <: DenseVector{T}} <: AbstractTensorMap{T, S, 2, 2}
    V1::S
    V2::S
    adjoint::Bool
    function BraidingTensor{T, S, A}(V1::S, V2::S, adjoint::Bool = false) where {T, S <: IndexSpace, A <: DenseVector{T}}
        for a in sectors(V1), b in sectors(V2), c in (a ⊗ b)
            Nsymbol(a, b, c) == Nsymbol(b, a, c) ||
                throw(ArgumentError("Cannot define a braiding between $a and $b"))
        end
        return new{T, S, A}(V1, V2, adjoint)
        # partial construction: only construct rowr and colr when needed
    end
end
function BraidingTensor{T}(V1::S, V2::S, adjoint::Bool = false) where {T, S <: IndexSpace}
    return braidingtensortype(S, T)(V1, V2, adjoint)
end
function BraidingTensor(V1::S, V2::S, adjoint::Bool = false) where {S <: IndexSpace}
    T = BraidingStyle(sectortype(S)) isa SymmetricBraiding ? Float64 : ComplexF64
    return BraidingTensor{T}(V1, V2, adjoint)
end
function BraidingTensor(V1::IndexSpace, V2::IndexSpace, adjoint::Bool = false)
    return BraidingTensor(promote(V1, V2)..., adjoint)
end
function BraidingTensor(V::HomSpace, adjoint::Bool = false)
    domain(V) == reverse(codomain(V)) ||
        throw(SpaceMismatch("Cannot define a braiding on $V"))
    return BraidingTensor(V[2], V[1], adjoint)
end
function BraidingTensor{T, S, A}(V::HomSpace, adjoint::Bool = false) where {T, S, A}
    domain(V) == reverse(codomain(V)) ||
        throw(SpaceMismatch("Cannot define a braiding on $V"))
    return BraidingTensor{T, S, A}(V[2], V[1], adjoint)
end
function BraidingTensor{T}(V::HomSpace, adjoint::Bool = false) where {T}
    domain(V) == reverse(codomain(V)) ||
        throw(SpaceMismatch("Cannot define a braiding on $V"))
    return BraidingTensor{T}(V[2], V[1], adjoint)
end

function Base.adjoint(b::BraidingTensor{T, S, A}) where {T, S, A}
    return BraidingTensor{T, S, A}(b.V1, b.V2, !b.adjoint)
end

# these are here to make the preprocessing for `@planar` expressions less painful
function braidingtensortype(::Type{S}, ::Type{TorA}) where {S <: IndexSpace, TorA}
    A = similarstoragetype(TorA)
    return BraidingTensor{scalartype(A), S, A}
end
braidingtensortype(V::S, ::Type{TorA}) where {S <: IndexSpace, TorA} = braidingtensortype(S, TorA)
braidingtensortype(V1::S, V2::S, ::Type{TorA}) where {S <: IndexSpace, TorA} = braidingtensortype(S, TorA)
function braidingtensortype(V1::IndexSpace, V2::IndexSpace, ::Type{TorA}) where {TorA}
    S = promote(V1, V2)
    return braidingtensortype(S..., TorA)
end
function braidingtensortype(V::HomSpace, ::Type{TorA}) where {TorA}
    return braidingtensortype(spacetype(V), TorA)
end

storagetype(::Type{BraidingTensor{T, S, A}}) where {T, S, A} = A
space(b::BraidingTensor) = b.adjoint ? b.V1 ⊗ b.V2 ← b.V2 ⊗ b.V1 : b.V2 ⊗ b.V1 ← b.V1 ⊗ b.V2

function Base.getindex(b::BraidingTensor)
    sectortype(b) === Trivial || throw(SectorMismatch())
    (V1, V2) = domain(b)
    d = (dim(V2), dim(V1), dim(V1), dim(V2))
    return sreshape(StridedView(block(b, Trivial())), d)
end

function _braiding_factor(f₁, f₂, inv::Bool = false)
    f₁.uncoupled == reverse(f₂.uncoupled) || return nothing
    I = sectortype(f₁)
    a, b = f₂.uncoupled
    c = f₂.coupled

    # braiding with unit is always possible
    # valid fusiontree pairs don't have to check Nsymbol(a, b, c)
    (isunit(a) || isunit(b)) && return one(sectorscalartype(I))

    BraidingStyle(I) isa NoBraiding && throw(SectorMismatch(lazy"Cannot braid sectors $a and $b"))

    if FusionStyle(I) isa MultiplicityFreeFusion
        r = inv ? conj(Rsymbol(b, a, c)) : Rsymbol(a, b, c)
    else
        Rmat = inv ? Rsymbol(b, a, c)' : Rsymbol(a, b, c)
        μ = only(f₂.vertices)
        ν = only(f₁.vertices)
        r = Rmat[μ, ν]
    end
    return r
end

# generates scalar indexing errors on GPU
function fill_braidingsubblock!(data, val)
    f(I) = ((I[1] == I[4]) & (I[2] == I[3])) * val
    return data .= f.(CartesianIndices(data))
end


@inline function subblock(
        b::BraidingTensor, (f₁, f₂)::Tuple{FusionTree{I, 2}, FusionTree{I, 2}}
    ) where {I <: Sector}
    I == sectortype(b) || throw(SectorMismatch())
    c = f₁.coupled
    V1, V2 = domain(b)
    @boundscheck begin
        c == f₂.coupled || throw(SectorMismatch())
        ((f₁.uncoupled[1] ∈ sectors(V2)) && (f₂.uncoupled[1] ∈ sectors(V1))) ||
            throw(SectorMismatch())
        ((f₁.uncoupled[2] ∈ sectors(V1)) && (f₂.uncoupled[2] ∈ sectors(V2))) ||
            throw(SectorMismatch())
    end
    d = (dims(codomain(b), f₁.uncoupled)..., dims(domain(b), f₂.uncoupled)...)
    data_parent = storagetype(b)(undef, prod(d))
    data = sreshape(StridedView(data_parent), d)
    r = _braiding_factor(f₁, f₂, b.adjoint)
    isnothing(r) ? zerovector!(data) : fill_braidingsubblock!(data, r)
    return data
end

# efficient copy constructor
Base.copy(b::BraidingTensor) = b

TensorMap(b::BraidingTensor) = copy!(similar(b), b)
Base.convert(::Type{TensorMap}, b::BraidingTensor) = TensorMap(b)

Base.complex(b::BraidingTensor{<:Complex}) = b
function Base.complex(b::BraidingTensor{T, S, A}) where {T, S, A}
    Tc = complex(T)
    Ac = similarstoragetype(A, Tc)
    return BraidingTensor{Tc, S, Ac}(space(b), b.adjoint)
end

# Trivial
function fill_braidingblock!(data, b::BraidingTensor, s::Trivial)
    V1, V2 = codomain(b)
    d1, d2 = dim(V1), dim(V2)
    subblock = sreshape(StridedView(data), (d1, d2, d2, d1))
    fill_braidingsubblock!(subblock, one(eltype(b)))
    return data
end

# Nontrivial
function fill_braidingblock!(data, b::BraidingTensor, s::Sector)
    base_offset = first(blockstructure(b)[s][2]) - 1

    for ((f₁, f₂), (sz, str, off)) in pairs(subblockstructure(space(b)))
        (f₁.coupled == f₂.coupled == s) || continue
        r = _braiding_factor(f₁, f₂, b.adjoint)
        # change offset to account for single block
        subblock = StridedView(data, sz, str, off - base_offset)
        isnothing(r) ? zerovector!(subblock) : fill_braidingsubblock!(subblock, r)
    end
    return data
end

function block(b::BraidingTensor, s::Sector)
    I = sectortype(b)
    I == typeof(s) || throw(SectorMismatch())

    # TODO: probably always square?
    m = blockdim(codomain(b), s)
    n = blockdim(domain(b), s)

    data = reshape(storagetype(b)(undef, m * n), (m, n))

    m * n == 0  && return data # s ∉ blocksectors(b)

    return fill_braidingblock!(data, b, s)
end

# Index manipulations
# -------------------
has_shared_permute(t::BraidingTensor, ::Index2Tuple) = false
function add_transform!(
        tdst::AbstractTensorMap,
        tsrc::BraidingTensor, (p₁, p₂)::Index2Tuple,
        fusiontreetransform,
        α::Number, β::Number, backend::AbstractBackend...
    )
    return add_transform!(
        tdst, TensorMap(tsrc), (p₁, p₂), fusiontreetransform, α, β,
        backend...
    )
end

function planarcontract!(
        C::AbstractTensorMap,
        A::BraidingTensor, pA::Index2Tuple,
        B::AbstractTensorMap, pB::Index2Tuple,
        pAB::Index2Tuple,
        α::Number, β::Number,
        backend, allocator
    )
    # special case only defined for contracting 2 indices
    length.(pA) == (2, 2) ||
        return planarcontract!(C, TensorMap(A), pA, B, pB, pAB, α, β, backend, allocator)

    spacecheck_contract(C, A, pA, false, B, pB, false, pAB)

    codA, domA = codomainind(A), domainind(A)
    codB, domB = codomainind(B), domainind(B)
    oindA, cindA, oindB, cindB = reorder_indices(
        codA, domA, codB, domB, pA..., reverse(pB)..., pAB...
    )

    I = sectortype(C)
    BraidingStyle(I) isa Bosonic &&
        return add_permute!(C, B, (reverse(cindB), oindB), α, β, backend)

    # Non-bosonic case: factor into a cyclic transpose (no crossings) + a single Artin braid
    # that swaps the two contracted legs, producing the R-symbol that A encodes. Naively
    # using a single `add_braid!` is wrong: it would resolve cyclic moves as crossings and
    # pick up spurious R-symbol factors.
    B_in_layout = (cindB == codB && oindB == domB)
    if B_in_layout
        B′ = B
    else
        B′ = TO.tensoralloc_add(
            scalartype(B), B, (cindB, oindB), false, Val(true), allocator
        )
        add_transpose!(B′, B, (cindB, oindB), One(), Zero(), backend)
    end

    levelsA = A.adjoint ? (1, 2, 2, 1) : (2, 1, 1, 2)
    N = numind(B)
    levels = (
        levelsA[cindA[1]], levelsA[cindA[2]],
        ntuple(Returns(3), N - 2)...,
    )

    add_braid!(
        C, B′, ((2, 1), ntuple(i -> i + 2, N - 2)),
        levels, α, β, backend,
    )

    B_in_layout || TO.tensorfree!(B′, allocator)
    return C
end
function planarcontract!(
        C::AbstractTensorMap,
        A::AbstractTensorMap, pA::Index2Tuple,
        B::BraidingTensor, pB::Index2Tuple,
        pAB::Index2Tuple,
        α::Number, β::Number,
        backend, allocator
    )
    # special case only defined for contracting all 4 indices of B (2 contracted + 2 open)
    length.(pB) == (2, 2) ||
        return planarcontract!(C, A, pA, TensorMap(B), pB, pAB, α, β, backend, allocator)

    spacecheck_contract(C, A, pA, false, B, pB, false, pAB)

    codA, domA = codomainind(A), domainind(A)
    codB, domB = codomainind(B), domainind(B)
    oindA, cindA, oindB, cindB = reorder_indices(
        codA, domA, codB, domB, pA..., reverse(pB)..., pAB...
    )

    I = sectortype(C)
    BraidingStyle(I) isa Bosonic &&
        return add_permute!(C, A, (oindA, reverse(cindA)), α, β, backend)

    # Non-bosonic case: cyclic transpose A → (oindA, cindA) (no crossings), then a single
    # Artin braid swaps A′'s last two indices, producing the R-symbol that B encodes. Naively
    # using a single `add_braid!` is wrong: it would resolve cyclic moves as crossings and
    # pick up spurious R-symbol factors.

    A_in_layout = (oindA == codA && cindA == domA)
    if A_in_layout
        A′ = A
    else
        A′ = TO.tensoralloc_add(
            scalartype(A), A, (oindA, cindA), false, Val(true), allocator
        )
        add_transpose!(A′, A, (oindA, cindA), One(), Zero(), backend)
    end

    levelsB = B.adjoint ? (1, 2, 2, 1) : (2, 1, 1, 2)
    N = numind(A)
    M = N - 2
    levels = (
        ntuple(Returns(3), M)...,
        levelsB[cindB[1]], levelsB[cindB[2]],
    )

    add_braid!(
        C, A′, (ntuple(identity, M), (N, N - 1)),
        levels, α, β, backend,
    )

    A_in_layout || TO.tensorfree!(A′, allocator)
    return C
end

# ambiguity fix:
function planarcontract!(
        C::AbstractTensorMap,
        A::BraidingTensor, pA::Index2Tuple,
        B::BraidingTensor, pB::Index2Tuple,
        pAB::Index2Tuple,
        α::Number, β::Number, backend, allocator
    )
    return planarcontract!(
        C, A, pA, TensorMap(B), pB, pAB, α, β, backend, allocator
    )
end
