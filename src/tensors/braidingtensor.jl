# BraidingTensor:
# special (2,2) tensor that implements a standard braiding operation
#====================================================================#
"""
    struct BraidingTensor{T,S<:IndexSpace} <: AbstractTensorMap{T, S, 2, 2}
    BraidingTensor(V1::S, V2::S, adjoint::Bool=false) where {S<:IndexSpace}

Specific subtype of [`AbstractTensorMap`](@ref) for representing the braiding tensor that
braids the first input over the second input; its inverse can be obtained as the adjoint.

It holds that `domain(BraidingTensor(V1, V2)) == V1 ⊗ V2` and
`codomain(BraidingTensor(V1, V2)) == V2 ⊗ V1`.
"""
struct BraidingTensor{T, S} <: AbstractTensorMap{T, S, 2, 2}
    V1::S
    V2::S
    adjoint::Bool
    function BraidingTensor{T, S}(V1::S, V2::S, adjoint::Bool = false) where {T, S <: IndexSpace}
        for a in sectors(V1)
            for b in sectors(V2)
                for c in (a ⊗ b)
                    Nsymbol(a, b, c) == Nsymbol(b, a, c) ||
                        throw(ArgumentError("Cannot define a braiding between $a and $b"))
                end
            end
        end
        return new{T, S}(V1, V2, adjoint)
        # partial construction: only construct rowr and colr when needed
    end
end
function BraidingTensor{T}(V1::S, V2::S, adjoint::Bool = false) where {T, S <: IndexSpace}
    return BraidingTensor{T, S}(V1, V2, adjoint)
end
function BraidingTensor{T}(V1::IndexSpace, V2::IndexSpace, adjoint::Bool = false) where {T}
    return BraidingTensor{T}(promote(V1, V2)..., adjoint)
end
function BraidingTensor(V1::IndexSpace, V2::IndexSpace, adjoint::Bool = false)
    return BraidingTensor(promote(V1, V2)..., adjoint)
end
function BraidingTensor(V1::S, V2::S, adjoint::Bool = false) where {S <: IndexSpace}
    T = BraidingStyle(sectortype(S)) isa SymmetricBraiding ? Float64 : ComplexF64
    return BraidingTensor{T, S}(V1, V2, adjoint)
end
function BraidingTensor(V::HomSpace, adjoint::Bool = false)
    domain(V) == reverse(codomain(V)) ||
        throw(SpaceMismatch("Cannot define a braiding on $V"))
    return BraidingTensor(V[2], V[1], adjoint)
end
function BraidingTensor{T}(V::HomSpace, adjoint::Bool = false) where {T}
    domain(V) == reverse(codomain(V)) ||
        throw(SpaceMismatch("Cannot define a braiding on $V"))
    return BraidingTensor{T}(V[2], V[1], adjoint)
end
function Base.adjoint(b::BraidingTensor{T, S}) where {T, S}
    return BraidingTensor{T, S}(b.V1, b.V2, !b.adjoint)
end

space(b::BraidingTensor) = b.adjoint ? b.V1 ⊗ b.V2 ← b.V2 ⊗ b.V1 : b.V2 ⊗ b.V1 ← b.V1 ⊗ b.V2

# specializations to ignore the storagetype of BraidingTensor
promote_storagetype(::Type{A}, ::Type{B}) where {A <: BraidingTensor, B <: AbstractTensorMap} = storagetype(B)
promote_storagetype(::Type{A}, ::Type{B}) where {A <: AbstractTensorMap, B <: BraidingTensor} = storagetype(A)
promote_storagetype(::Type{A}, ::Type{B}) where {A <: BraidingTensor, B <: BraidingTensor} = storagetype(A)

promote_storagetype(::Type{T}, ::Type{A}, ::Type{B}) where {T <: Number, A <: BraidingTensor, B <: AbstractTensorMap} =
    similarstoragetype(B, T)
promote_storagetype(::Type{T}, ::Type{A}, ::Type{B}) where {T <: Number, A <: AbstractTensorMap, B <: BraidingTensor} =
    similarstoragetype(A, T)
promote_storagetype(::Type{T}, ::Type{A}, ::Type{B}) where {T <: Number, A <: BraidingTensor, B <: BraidingTensor} =
    similarstoragetype(A, T)

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
    n1 = d[1] * d[2]
    n2 = d[3] * d[4]
    data = sreshape(StridedView(Matrix{eltype(b)}(undef, n1, n2)), d)
    fill!(data, zero(eltype(b)))

    r = _braiding_factor(f₁, f₂, b.adjoint)
    if !isnothing(r)
        @inbounds for i in axes(data, 1), j in axes(data, 2)
            data[i, j, j, i] = r
        end
    end
    return data
end

# efficient copy constructor
Base.copy(b::BraidingTensor) = b

TensorMap(b::BraidingTensor) = copy!(similar(b), b)
Base.convert(::Type{TensorMap}, b::BraidingTensor) = TensorMap(b)

Base.complex(b::BraidingTensor{<:Complex}) = b
function Base.complex(b::BraidingTensor)
    return BraidingTensor{complex(scalartype(b))}(space(b), b.adjoint)
end

function block(b::BraidingTensor, s::Sector)
    I = sectortype(b)
    I == typeof(s) || throw(SectorMismatch())

    # TODO: probably always square?
    m = blockdim(codomain(b), s)
    n = blockdim(domain(b), s)
    data = Matrix{eltype(b)}(undef, (m, n))

    length(data) == 0 && return data # s ∉ blocksectors(b)

    data = fill!(data, zero(eltype(b)))

    V1, V2 = codomain(b)
    if sectortype(b) === Trivial
        d1, d2 = dim(V1), dim(V2)
        subblock = sreshape(StridedView(data), (d1, d2, d2, d1))
        @inbounds for i in axes(subblock, 1), j in axes(subblock, 2)
            subblock[i, j, j, i] = one(eltype(b))
        end
        return data
    end

    base_offset = first(blockstructure(b)[s][2]) - 1

    for ((f₁, f₂), (sz, str, off)) in pairs(subblockstructure(space(b)))
        (f₁.coupled == f₂.coupled == s) || continue
        r = _braiding_factor(f₁, f₂, b.adjoint)
        isnothing(r) && continue
        # change offset to account for single block
        subblock = StridedView(data, sz, str, off - base_offset)
        @inbounds for i in axes(subblock, 1), j in axes(subblock, 2)
            subblock[i, j, j, i] = r
        end
    end

    return data
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
