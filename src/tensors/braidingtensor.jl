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

# TODO: this will probably give issues with GPUs, so we should try to avoid
# calling this method alltogether
storagetype(::Type{BraidingTensor{T, S}}) where {T, S} = Vector{T}

function Base.getindex(b::BraidingTensor)
    sectortype(b) === Trivial || throw(SectorMismatch())
    (V1, V2) = domain(b)
    d = (dim(V2), dim(V1), dim(V1), dim(V2))
    return sreshape(StridedView(block(b, Trivial())), d)
end

@inline function Base.getindex(
        b::BraidingTensor, f₁::FusionTree{I, 2}, f₂::FusionTree{I, 2}
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
    @inbounds begin
        d = (dims(codomain(b), f₁.uncoupled)..., dims(domain(b), f₂.uncoupled)...)
        n1 = d[1] * d[2]
        n2 = d[3] * d[4]
        data = sreshape(StridedView(Matrix{eltype(b)}(undef, n1, n2)), d)
        fill!(data, zero(eltype(b)))
        if f₁.uncoupled == reverse(f₂.uncoupled)
            braiddict = artin_braid(f₂, 1; inv = b.adjoint)
            r = get(braiddict, f₁, zero(valtype(braiddict)))
            @inbounds for i in axes(data, 1), j in axes(data, 2)
                data[i, j, j, i] = r
            end
        end
        return data
    end
end
@inline function Base.getindex(b::BraidingTensor, ::Nothing, ::Nothing)
    sectortype(b) === Trivial || throw(SectorMismatch())
    return getindex(b)
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
    sectortype(b) == typeof(s) || throw(SectorMismatch())

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

    structure = fusionblockstructure(b)
    base_offset = first(structure.blockstructure[s][2]) - 1

    for ((f1, f2), (sz, str, off)) in
        zip(structure.fusiontreelist, structure.fusiontreestructure)
        if (f1.uncoupled != reverse(f2.uncoupled)) || !(f1.coupled == f2.coupled == s)
            continue
        end

        braiddict = artin_braid(f2, 1; inv = b.adjoint)
        haskey(braiddict, f1) || continue
        r = braiddict[f1]

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

# VectorInterface
# ---------------
# TODO

# TensorOperations
# ----------------
# TODO: implement specialized methods

function TO.tensoradd!(
        C::AbstractTensorMap,
        A::BraidingTensor, pA::Index2Tuple, conjA::Symbol,
        α::Number, β::Number, backend = TO.DefaultBackend(), allocator = TO.DefaultAllocator()
    )
    return TO.tensoradd!(C, TensorMap(A), pA, conjA, α, β, backend, allocator)
end

# Planar operations
# -----------------
# TODO: implement specialized methods

function planaradd!(
        C::AbstractTensorMap,
        A::BraidingTensor, p::Index2Tuple,
        α::Number, β::Number, backend, allocator
    )
    return planaradd!(C, TensorMap(A), p, α, β, backend, allocator)
end

function planarcontract!(
        C::AbstractTensorMap,
        A::BraidingTensor,
        (oindA, cindA)::Index2Tuple,
        B::AbstractTensorMap,
        (cindB, oindB)::Index2Tuple,
        (p1, p2)::Index2Tuple,
        α::Number, β::Number,
        backend, allocator
    )
    # special case only defined for contracting 2 indices
    length(oindA) == length(cindA) == 2 ||
        return planarcontract!(
        C, TensorMap(A), (oindA, cindA), B, (cindB, oindB), (p1, p2),
        α, β, backend, allocator
    )

    codA, domA = codomainind(A), domainind(A)
    codB, domB = codomainind(B), domainind(B)
    oindA, cindA, oindB, cindB = reorder_indices(
        codA, domA, codB, domB, oindA, cindA, oindB, cindB, p1, p2
    )

    if space(B, cindB[1]) != space(A, cindA[1])' ||
            space(B, cindB[2]) != space(A, cindA[2])'
        throw(SpaceMismatch("$(space(C)) ≠ permute($(space(A))[$oindA, $cindA] * $(space(B))[$cindB, $oindB], ($p1, $p2)"))
    end

    if BraidingStyle(sectortype(B)) isa Bosonic
        return add_permute!(C, B, (reverse(cindB), oindB), α, β, backend)
    end

    τ_levels = A.adjoint ? (1, 2, 2, 1) : (2, 1, 1, 2)
    scale!(C, β)

    inv_braid = τ_levels[cindA[1]] > τ_levels[cindA[2]]
    for (f₁, f₂) in fusiontrees(B)
        local newtrees
        for ((f₁′, f₂′), coeff′) in transpose(f₁, f₂, cindB, oindB)
            for (f₁′′, coeff′′) in artin_braid(f₁′, 1; inv = inv_braid)
                f12 = (f₁′′, f₂′)
                coeff = coeff′ * coeff′′
                if @isdefined newtrees
                    newtrees[f12] = get(newtrees, f12, zero(coeff)) + coeff
                else
                    newtrees = Dict(f12 => coeff)
                end
            end
        end
        for ((f₁′, f₂′), coeff) in newtrees
            TO.tensoradd!(
                C[f₁′, f₂′], B[f₁, f₂], (reverse(cindB), oindB), false,
                α * coeff, One(), backend, allocator
            )
        end
    end
    return C
end
function planarcontract!(
        C::AbstractTensorMap,
        A::AbstractTensorMap, (oindA, cindA)::Index2Tuple,
        B::BraidingTensor, (cindB, oindB)::Index2Tuple,
        (p1, p2)::Index2Tuple,
        α::Number, β::Number,
        backend, allocator
    )
    # special case only defined for contracting 2 indices
    length(oindB) == length(cindB) == 2 ||
        return planarcontract!(
        C, A, (oindA, cindA), TensorMap(B), (cindB, oindB), (p1, p2),
        α, β, backend, allocator
    )

    codA, domA = codomainind(A), domainind(A)
    codB, domB = codomainind(B), domainind(B)
    oindA, cindA, oindB, cindB = reorder_indices(
        codA, domA, codB, domB, oindA, cindA, oindB, cindB, p1, p2
    )

    if space(B, cindB[1]) != space(A, cindA[1])' || space(B, cindB[2]) != space(A, cindA[2])'
        throw(SpaceMismatch("$(space(C)) ≠ permute($(space(A))[$oindA, $cindA] * $(space(B))[$cindB, $oindB], ($p1, $p2)"))
    end

    if BraidingStyle(sectortype(A)) isa Bosonic
        return add_permute!(C, A, (oindA, reverse(cindA)), α, β, backend)
    end

    scale!(C, β)
    τ_levels = B.adjoint ? (1, 2, 2, 1) : (2, 1, 1, 2)
    inv_braid = τ_levels[cindB[1]] > τ_levels[cindB[2]]

    for (f₁, f₂) in fusiontrees(A)
        local newtrees
        for ((f₁′, f₂′), coeff′) in transpose(f₁, f₂, oindA, cindA)
            for (f₂′′, coeff′′) in artin_braid(f₂′, 1; inv = inv_braid)
                f12 = (f₁′, f₂′′)
                coeff = coeff′ * conj(coeff′′)
                if @isdefined newtrees
                    newtrees[f12] = get(newtrees, f12, zero(coeff)) + coeff
                else
                    newtrees = Dict(f12 => coeff)
                end
            end
        end
        for ((f₁′, f₂′), coeff) in newtrees
            TO.tensoradd!(
                C[f₁′, f₂′], A[f₁, f₂], (oindA, reverse(cindA)), false, α * coeff,
                One(), backend, allocator
            )
        end
    end
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
        C, TensorMap(A), pA, TensorMap(B), pB, pAB, α, β, backend, allocator
    )
end

function planartrace!(
        C::AbstractTensorMap,
        A::BraidingTensor,
        p::Index2Tuple, q::Index2Tuple,
        α::Number, β::Number,
        backend, allocator
    )
    return planartrace!(C, TensorMap(A), p, q, α, β, backend, allocator)
end

# function planarcontract!(C::AbstractTensorMap{S,N₁,N₂},
#                          A::BraidingTensor{S},
#                          (oindA, cindA)::Index2Tuple{0,4},
#                          B::AbstractTensorMap{S},
#                          (cindB, oindB)::Index2Tuple{4,<:Any},
#                          (p1, p2)::Index2Tuple{N₁,N₂},
#                          α::Number, β::Number,
#                          backend::Backend...) where {S,N₁,N₂}
#     codA, domA = codomainind(A), domainind(A)
#     codB, domB = codomainind(B), domainind(B)
#     oindA, cindA, oindB, cindB = reorder_indices(codA, domA, codB, domB, oindA, cindA,
#                                                  oindB, cindB, p1, p2)

#     @assert space(B, cindB[1]) == space(A, cindA[1])' &&
#             space(B, cindB[2]) == space(A, cindA[2])' &&
#             space(B, cindB[3]) == space(A, cindA[3])' &&
#             space(B, cindB[4]) == space(A, cindA[4])'

#     if BraidingStyle(sectortype(B)) isa Bosonic
#         return trace!(α, B, β, C, (), oindB, (cindB[1], cindB[2]), (cindB[3], cindB[4]))
#     end

#     if iszero(β)
#         fill!(C, β)
#     elseif β != 1
#         rmul!(C, β)
#     end
#     I = sectortype(B)
#     u = one(I)
#     f₀ = FusionTree{I}((), u, (), (), ())
#     braidingtensor_levels = A.adjoint ? (1, 2, 2, 1) : (2, 1, 1, 2)
#     inv_braid = braidingtensor_levels[cindA[2]] > braidingtensor_levels[cindA[3]]
#     for (f₁, f₂) in fusiontrees(B)
#         local newtrees
#         for ((f₁′, f₂′), coeff′) in transpose(f₁, f₂, cindB, oindB)
#             f₁′.coupled == u || continue
#             a = f₁′.uncoupled[1]
#             b = f₁′.uncoupled[2]
#             f₁′.uncoupled[3] == dual(a) || continue
#             f₁′.uncoupled[4] == dual(b) || continue
#             # should be automatic by matching spaces:
#             # f₁′.isdual[1] != f₁′.isdual[3] || continue
#             # f₁′.isdual[2] != f₁′.isdual[4] || continue
#             for (f₁′′, coeff′′) in artin_braid(f₁′, 2; inv=inv_braid)
#                 f₁′′.innerlines[1] == u || continue
#                 coeff = coeff′ * coeff′′ * sqrtdim(a) * sqrtdim(b)
#                 if f₁′′.isdual[1]
#                     coeff *= frobeniusschur(a)
#                 end
#                 if f₁′′.isdual[3]
#                     coeff *= frobeniusschur(b)
#                 end
#                 f12 = (f₀, f₂′)
#                 if @isdefined newtrees
#                     newtrees[f12] = get(newtrees, f12, zero(coeff)) + coeff
#                 else
#                     newtrees = Dict(f12 => coeff)
#                 end
#             end
#         end
#         @isdefined(newtrees) || continue
#         for ((f₁′, f₂′), coeff) in newtrees
#             TO._trace!(coeff * α, B[f₁, f₂], true, C[f₁′, f₂′], oindB,
#                        (cindB[1], cindB[2]), (cindB[3], cindB[4]))
#         end
#     end
#     return C
# end
# function planarcontract!(C::AbstractTensorMap{S,N₁,N₂},
#                          A::AbstractTensorMap{S},
#                          (oindA, cindA)::Index2Tuple{0,4},
#                          B::BraidingTensor{S},
#                          (cindB, oindB)::Index2Tuple{4,<:Any},
#                          (p1, p2)::Index2Tuple{N₁,N₂},
#                          α::Number, β::Number,
#                          backends...) where {S,N₁,N₂}
#     codA, domA = codomainind(A), domainind(A)
#     codB, domB = codomainind(B), domainind(B)
#     oindA, cindA, oindB, cindB = reorder_indices(codA, domA, codB, domB, oindA, cindA,
#                                                  oindB, cindB, p1, p2)

#     @assert space(B, cindB[1]) == space(A, cindA[1])' &&
#             space(B, cindB[2]) == space(A, cindA[2])' &&
#             space(B, cindB[3]) == space(A, cindA[3])' &&
#             space(B, cindB[4]) == space(A, cindA[4])'

#     if BraidingStyle(sectortype(B)) isa Bosonic
#         return trace!(α, A, β, C, oindA, (), (cindA[1], cindA[2]), (cindA[3], cindA[4]))
#     end

#     if iszero(β)
#         fill!(C, β)
#     elseif β != 1
#         rmul!(C, β)
#     end
#     I = sectortype(B)
#     u = one(I)
#     f₀ = FusionTree{I}((), u, (), (), ())
#     braidingtensor_levels = B.adjoint ? (1, 2, 2, 1) : (2, 1, 1, 2)
#     inv_braid = braidingtensor_levels[cindB[2]] > braidingtensor_levels[cindB[3]]
#     for (f₁, f₂) in fusiontrees(A)
#         local newtrees
#         for ((f₁′, f₂′), coeff′) in transpose(f₁, f₂, oindA, cindA)
#             f₂′.coupled == u || continue
#             a = f₂′.uncoupled[1]
#             b = f₂′.uncoupled[2]
#             f₂′.uncoupled[3] == dual(a) || continue
#             f₂′.uncoupled[4] == dual(b) || continue
#             # should be automatic by matching spaces:
#             # f₂′.isdual[1] != f₂′.isdual[3] || continue
#             # f₂′.isdual[3] != f₂′.isdual[4] || continue
#             for (f₂′′, coeff′′) in artin_braid(f₂′, 2; inv=inv_braid)
#                 f₂′′.innerlines[1] == u || continue
#                 coeff = coeff′ * conj(coeff′′ * sqrtdim(a) * sqrtdim(b))
#                 if f₂′′.isdual[1]
#                     coeff *= conj(frobeniusschur(a))
#                 end
#                 if f₂′′.isdual[3]
#                     coeff *= conj(frobeniusschur(b))
#                 end
#                 f12 = (f₁′, f₀)
#                 if @isdefined newtrees
#                     newtrees[f12] = get(newtrees, f12, zero(coeff)) + coeff
#                 else
#                     newtrees = Dict(f12 => coeff)
#                 end
#             end
#         end
#         @isdefined(newtrees) || continue
#         for ((f₁′, f₂′), coeff) in newtrees
#             TO._trace!(coeff * α, A[f₁, f₂], true, C[f₁′, f₂′], oindA,
#                        (cindA[1], cindA[2]), (cindA[3], cindA[4]))
#         end
#     end
#     return C
# end
# function planarcontract!(C::AbstractTensorMap{S,N₁,N₂},
#                          A::BraidingTensor{S},
#                          (oindA, cindA)::Index2Tuple{1,3},
#                          B::AbstractTensorMap{S},
#                          (cindB, oindB)::Index2Tuple{1,<:Any},
#                          (p1, p2)::Index2Tuple{N₁,N₂},
#                          α::Number, β::Number,
#                          backend::Backend...) where {S,N₁,N₂}
#     codA, domA = codomainind(A), domainind(A)
#     codB, domB = codomainind(B), domainind(B)
#     oindA, cindA, oindB, cindB = reorder_indices(codA, domA, codB, domB, oindA, cindA,
#                                                  oindB, cindB, p1, p2)

#     @assert space(B, cindB[1]) == space(A, cindA[1])' &&
#             space(B, cindB[2]) == space(A, cindA[2])' &&
#             space(B, cindB[3]) == space(A, cindA[3])'

#     if BraidingStyle(sectortype(B)) isa Bosonic
#         return trace!(α, B, β, C, (cindB[2],), oindB, (cindB[1],), (cindB[3],))
#     end

#     if iszero(β)
#         fill!(C, β)
#     elseif β != 1
#         rmul!(C, β)
#     end
#     I = sectortype(B)
#     u = one(I)
#     braidingtensor_levels = A.adjoint ? (1, 2, 2, 1) : (2, 1, 1, 2)
#     inv_braid = braidingtensor_levels[cindA[2]] > braidingtensor_levels[cindA[3]]
#     for (f₁, f₂) in fusiontrees(B)
#         local newtrees
#         for ((f₁′, f₂′), coeff′) in transpose(f₁, f₂, cindB, oindB)
#             a = f₁′.uncoupled[1]
#             b = f₁′.uncoupled[2]
#             b == f₁′.coupled || continue
#             a == dual(f₁′.uncoupled[3]) || continue
#             # should be automatic by matching spaces:
#             # f₁′.isdual[1] != f₁.isdual[3] || continue
#             for (f₁′′, coeff′′) in artin_braid(f₁′, 2; inv=inv_braid)
#                 f₁′′.innerlines[1] == u || continue
#                 coeff = coeff′ * coeff′′ * sqrtdim(a)
#                 if f₁′′.isdual[1]
#                     coeff *= frobeniusschur(a)
#                 end
#                 f₁′′′ = FusionTree{I}((b,), b, (f₁′′.isdual[3],), (), ())
#                 f12 = (f₁′′′, f₂′)
#                 if @isdefined newtrees
#                     newtrees[f12] = get(newtrees, f12, zero(coeff)) + coeff
#                 else
#                     newtrees = Dict(f12 => coeff)
#                 end
#             end
#         end
#         @isdefined(newtrees) || continue
#         for ((f₁′, f₂′), coeff) in newtrees
#             TO._trace!(coeff * α, B[f₁, f₂], true, C[f₁′, f₂′],
#                        (cindB[2], oindB...), (cindB[1],), (cindB[3],))
#         end
#     end
#     return C
# end
# function planarcontract!(C::AbstractTensorMap{S,N₁,N₂},
#                          A::AbstractTensorMap{S},
#                          (oindA, cindA)::Index2Tuple{<:Any,3},
#                          B::BraidingTensor{S},
#                          (cindB, oindB)::Index2Tuple{3,1},
#                          (p1, p2)::Index2Tuple{N₁,N₂},
#                          α::Number, β::Number,
#                          backend::Backend...) where {S,N₁,N₂}
#     codA, domA = codomainind(A), domainind(A)
#     codB, domB = codomainind(B), domainind(B)
#     oindA, cindA, oindB, cindB = reorder_indices(codA, domA, codB, domB, oindA, cindA,
#                                                  oindB, cindB, p1, p2)

#     @assert space(B, cindB[1]) == space(A, cindA[1])' &&
#             space(B, cindB[2]) == space(A, cindA[2])' &&
#             space(B, cindB[3]) == space(A, cindA[3])'

#     if BraidingStyle(sectortype(A)) isa Bosonic
#         return trace!(α, A, β, C, oindA, (cindA[2],), (cindA[1],), (cindA[3],))
#     end

#     if iszero(β)
#         fill!(C, β)
#     elseif β != 1
#         rmul!(C, β)
#     end
#     I = sectortype(B)
#     u = one(I)
#     braidingtensor_levels = B.adjoint ? (1, 2, 2, 1) : (2, 1, 1, 2)
#     inv_braid = braidingtensor_levels[cindB[2]] > braidingtensor_levels[cindB[3]]
#     for (f₁, f₂) in fusiontrees(A)
#         local newtrees
#         for ((f₁′, f₂′), coeff′) in transpose(f₁, f₂, oindA, cindA)
#             a = f₂′.uncoupled[1]
#             b = f₂′.uncoupled[2]
#             b == f₂′.coupled || continue
#             a == dual(f₂′.uncoupled[3]) || continue
#             # should be automatic by matching spaces:
#             # f₂′.isdual[1] != f₂.isdual[3] || continue
#             for (f₂′′, coeff′′) in artin_braid(f₂′, 2; inv=inv_braid)
#                 f₂′′.innerlines[1] == u || continue
#                 coeff = coeff′ * conj(coeff′′ * sqrtdim(a))
#                 if f₂′′.isdual[1]
#                     coeff *= conj(frobeniusschur(a))
#                 end
#                 f₂′′′ = FusionTree{I}((b,), b, (f₂′′.isdual[3],), (), ())
#                 f12 = (f₁′, f₂′′′)
#                 if @isdefined newtrees
#                     newtrees[f12] = get(newtrees, f12, zero(coeff)) + coeff
#                 else
#                     newtrees = Dict(f12 => coeff)
#                 end
#             end
#         end
#         @isdefined(newtrees) || continue
#         for ((f₁′, f₂′), coeff) in newtrees
#             TO._trace!(coeff * α, A[f₁, f₂], true, C[f₁′, f₂′],
#                        (oindA..., cindA[2]), (cindA[1],), (cindA[3],))
#         end
#     end
#     return C
# end
