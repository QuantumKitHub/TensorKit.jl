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
        for ((f₁′, f₂′), coeff′) in transpose((f₁, f₂), (cindB, oindB))
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

    p = (oindA, reverse(cindA))
    N = length(oindA)
    levels = (ntuple(identity, N)..., (B.adjoint ? (N + 1, N + 2) : (N + 2, N + 1))...)
    return add_braid!(C, A, p, levels, α, β, backend)
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

# function planarcontract!(C::AbstractTensorMap{<:Any,S,N₁,N₂},
#                          A::BraidingTensor{S},
#                          (oindA, cindA)::Index2Tuple{0,4},
#                          B::AbstractTensorMap{<:Any,S},
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
#     u = unit(I)
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
#                     coeff *= frobenius_schur_phase(a)
#                 end
#                 if f₁′′.isdual[3]
#                     coeff *= frobenius_schur_phase(b)
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
# function planarcontract!(C::AbstractTensorMap{<:Any,S,N₁,N₂},
#                          A::AbstractTensorMap{<:Any,S},
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
#     u = unit(I)
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
#                     coeff *= conj(frobenius_schur_phase(a))
#                 end
#                 if f₂′′.isdual[3]
#                     coeff *= conj(frobenius_schur_phase(b))
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
# function planarcontract!(C::AbstractTensorMap{<:Any,S,N₁,N₂},
#                          A::BraidingTensor{S},
#                          (oindA, cindA)::Index2Tuple{1,3},
#                          B::AbstractTensorMap{<:Any,S},
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
#     u = unit(I)
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
#                     coeff *= frobenius_schur_phase(a)
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
# function planarcontract!(C::AbstractTensorMap{<:Any,S,N₁,N₂},
#                          A::AbstractTensorMap{<:Any,S},
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
#     u = unit(I)
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
#                     coeff *= conj(frobenius_schur_phase(a))
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
