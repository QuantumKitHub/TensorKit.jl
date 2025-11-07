# Implement full TensorOperations.jl interface
#----------------------------------------------
TO.tensorstructure(t::AbstractTensorMap) = space(t)
function TO.tensorstructure(t::AbstractTensorMap, iA::Int, conjA::Bool)
    return !conjA ? space(t, iA) : conj(space(t, iA))
end

function TO.tensoralloc(
        ::Type{TT}, structure::TensorMapSpace{S, N₁, N₂},
        istemp::Val, allocator = TO.DefaultAllocator()
    ) where {T, S, N₁, N₂, TT <: AbstractTensorMap{T, S, N₁, N₂}}
    A = storagetype(TT)
    dim = fusionblockstructure(structure).totaldim
    data = TO.tensoralloc(A, dim, istemp, allocator)
    # return TT(data, structure)
    return TensorMap{T}(data, structure)
end

function TO.tensorfree!(t::TensorMap, allocator = TO.DefaultAllocator())
    TO.tensorfree!(t.data, allocator)
    return nothing
end

TO.tensorscalar(t::AbstractTensorMap) = scalar(t)

function _canonicalize(
        p::Index2Tuple{N₁, N₂}, ::AbstractTensorMap{<:IndexSpace, N₁, N₂}
    ) where {N₁, N₂}
    return p
end
_canonicalize(p::Index2Tuple, t::AbstractTensorMap) = _canonicalize(linearize(p), t)
function _canonicalize(p::IndexTuple, t::AbstractTensorMap)
    p₁ = TupleTools.getindices(p, codomainind(t))
    p₂ = TupleTools.getindices(p, domainind(t))
    return (p₁, p₂)
end

# tensoradd!
function TO.tensoradd!(
        C::AbstractTensorMap,
        A::AbstractTensorMap, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend, allocator
    )
    if conjA
        A′ = adjoint(A)
        pA′ = adjointtensorindices(A, _canonicalize(pA, C))
        add_permute!(C, A′, pA′, α, β, backend)
    else
        add_permute!(C, A, _canonicalize(pA, C), α, β, backend)
    end
    return C
end

function TO.tensoradd_type(
        TC, A::AbstractTensorMap, ::Index2Tuple{N₁, N₂}, ::Bool
    ) where {N₁, N₂}
    I = sectortype(A)
    M = similarstoragetype(A, sectorscalartype(I) <: Real ? TC : complex(TC))
    return tensormaptype(spacetype(A), N₁, N₂, M)
end

function TO.tensoradd_structure(
        A::AbstractTensorMap, pA::Index2Tuple{N₁, N₂}, conjA::Bool
    ) where {N₁, N₂}
    if !conjA
        # don't use `permute` as this is also used when indices are traced
        return select(space(A), pA)
    else
        return TO.tensoradd_structure(adjoint(A), adjointtensorindices(A, pA), false)
    end
end

# tensortrace!
function TO.tensortrace!(
        C::AbstractTensorMap,
        A::AbstractTensorMap, p::Index2Tuple, q::Index2Tuple,
        conjA::Bool,
        α::Number, β::Number, backend, allocator
    )
    if conjA
        A′ = adjoint(A)
        p′ = adjointtensorindices(A, _canonicalize(p, C))
        q′ = adjointtensorindices(A, q)
        trace_permute!(C, A′, p′, q′, α, β, backend)
    else
        trace_permute!(C, A, _canonicalize(p, C), q, α, β, backend)
    end
    return C
end

# tensorcontract!
function TO.tensorcontract!(
        C::AbstractTensorMap,
        A::AbstractTensorMap, pA::Index2Tuple, conjA::Bool,
        B::AbstractTensorMap, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple, α::Number, β::Number,
        backend, allocator
    )
    pAB′ = _canonicalize(pAB, C)
    if conjA && conjB
        A′ = A'
        pA′ = adjointtensorindices(A, pA)
        B′ = B'
        pB′ = adjointtensorindices(B, pB)
        contract!(C, A′, pA′, B′, pB′, pAB′, α, β, backend, allocator)
    elseif conjA
        A′ = A'
        pA′ = adjointtensorindices(A, pA)
        contract!(C, A′, pA′, B, pB, pAB′, α, β, backend, allocator)
    elseif conjB
        B′ = B'
        pB′ = adjointtensorindices(B, pB)
        contract!(C, A, pA, B′, pB′, pAB′, α, β, backend, allocator)
    else
        contract!(C, A, pA, B, pB, pAB′, α, β, backend, allocator)
    end
    return C
end

function TO.tensorcontract_type(
        TC,
        A::AbstractTensorMap, ::Index2Tuple, ::Bool,
        B::AbstractTensorMap, ::Index2Tuple, ::Bool,
        ::Index2Tuple{N₁, N₂}
    ) where {N₁, N₂}
    spacetype(A) == spacetype(B) || throw(SpaceMismatch("incompatible space types"))
    I = sectortype(A)
    M = similarstoragetype(A, sectorscalartype(I) <: Real ? TC : complex(TC))
    MB = similarstoragetype(B, sectorscalartype(I) <: Real ? TC : complex(TC))
    M == MB || throw(ArgumentError("incompatible storage types:\n$(M) ≠ $(MB)"))
    return tensormaptype(spacetype(A), N₁, N₂, M)
end

function TO.tensorcontract_structure(
        A::AbstractTensorMap, pA::Index2Tuple, conjA::Bool,
        B::AbstractTensorMap, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple{N₁, N₂}
    ) where {N₁, N₂}
    sA = TO.tensoradd_structure(A, pA, conjA)
    sB = TO.tensoradd_structure(B, pB, conjB)
    return permute(compose(sA, sB), pAB)
end

function TO.checkcontractible(
        tA::AbstractTensorMap, iA::Int, conjA::Bool,
        tB::AbstractTensorMap, iB::Int, conjB::Bool,
        label
    )
    sA = TO.tensorstructure(tA, iA, conjA)'
    sB = TO.tensorstructure(tB, iB, conjB)
    sA == sB ||
        throw(SpaceMismatch("incompatible spaces for $label: $sA ≠ $sB"))
    return nothing
end

TO.tensorcost(t::AbstractTensorMap, i::Int) = dim(space(t, i))

#----------------
# IMPLEMENTATONS
#----------------

# Trace implementation
#----------------------
"""
    trace_permute!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap,
                   (p₁, p₂)::Index2Tuple, (q₁, q₂)::Index2Tuple,
                   α::Number, β::Number, backend=TO.DefaultBackend())

Return the updated `tdst`, which is the result of adding `α * tsrc` to `tdst` after permuting
the indices of `tsrc` according to `(p₁, p₂)` and furthermore tracing the indices in `q₁` and `q₂`.
"""
function trace_permute!(
        tdst::AbstractTensorMap,
        tsrc::AbstractTensorMap,
        (p₁, p₂)::Index2Tuple,
        (q₁, q₂)::Index2Tuple,
        α::Number,
        β::Number,
        backend = TO.DefaultBackend()
    )
    # some input checks
    (S = spacetype(tdst)) == spacetype(tsrc) ||
        throw(SpaceMismatch("incompatible spacetypes"))
    if !(BraidingStyle(sectortype(S)) isa SymmetricBraiding)
        throw(SectorMismatch("only tensors with symmetric braiding rules can be contracted; try `@planar` instead"))
    end
    (N₃ = length(q₁)) == length(q₂) ||
        throw(IndexError("number of trace indices does not match"))

    N₁, N₂ = length(p₁), length(p₂)

    @boundscheck begin
        space(tdst) == select(space(tsrc), (p₁, p₂)) ||
            throw(SpaceMismatch("trace: tsrc = $(codomain(tsrc))←$(domain(tsrc)),
                    tdst = $(codomain(tdst))←$(domain(tdst)), p₁ = $(p₁), p₂ = $(p₂)"))
        all(i -> space(tsrc, q₁[i]) == dual(space(tsrc, q₂[i])), 1:N₃) ||
            throw(SpaceMismatch("trace: tsrc = $(codomain(tsrc))←$(domain(tsrc)),
                    q₁ = $(q₁), q₂ = $(q₂)"))
    end

    I = sectortype(S)
    # TODO: is it worth treating UniqueFusion separately? Is it worth to add multithreading support?
    if I === Trivial
        cod = codomain(tsrc)
        dom = domain(tsrc)
        n = length(cod)
        TO.tensortrace!(tdst[], tsrc[], (p₁, p₂), (q₁, q₂), false, α, β, backend)
        # elseif FusionStyle(I) isa UniqueFusion
    else
        cod = codomain(tsrc)
        dom = domain(tsrc)
        n = length(cod)
        scale!(tdst, β)
        r₁ = (p₁..., q₁...)
        r₂ = (p₂..., q₂...)
        for (f₁, f₂) in fusiontrees(tsrc)
            for ((f₁′, f₂′), coeff) in permute(f₁, f₂, r₁, r₂)
                f₁′′, g₁ = split(f₁′, N₁)
                f₂′′, g₂ = split(f₂′, N₂)
                g₁ == g₂ || continue
                coeff *= dim(g₁.coupled) / dim(g₁.uncoupled[1])
                for i in 2:length(g₁.uncoupled)
                    if !(g₁.isdual[i])
                        coeff *= twist(g₁.uncoupled[i])
                    end
                end
                C = tdst[f₁′′, f₂′′]
                A = tsrc[f₁, f₂]
                α′ = α * coeff
                TO.tensortrace!(C, A, (p₁, p₂), (q₁, q₂), false, α′, One(), backend)
            end
        end
    end
    return tdst
end

# Contract implementation
#-------------------------
# TODO: contraction with either A or B a rank (1, 1) tensor does not require to
# permute the fusion tree and should therefore be special cased. This will speed
# up MPS algorithms
""" 
    contract!(C::AbstractTensorMap,
              A::AbstractTensorMap, (oindA, cindA)::Index2Tuple,
              B::AbstractTensorMap, (cindB, oindB)::Index2Tuple,
              (p₁, p₂)::Index2Tuple,
              α::Number, β::Number,
              backend, allocator)

Return the updated `C`, which is the result of adding `α * A * B` to `C` after permuting
the indices of `A` and `B` according to `(oindA, cindA)` and `(cindB, oindB)` respectively.
"""
function contract!(
        C::AbstractTensorMap,
        A::AbstractTensorMap, pA::Index2Tuple,
        B::AbstractTensorMap, pB::Index2Tuple,
        pAB::Index2Tuple, α::Number, β::Number,
        backend, allocator
    )
    length(pA[2]) == length(pB[1]) ||
        throw(IndexError("number of contracted indices does not match"))
    N₁, N₂ = length(pA[1]), length(pB[2])

    # find optimal contraction scheme by checking the following options:
    # - sorting the contracted inds of A or B to avoid permutations
    # - contracting B with A instead to avoid permutations

    qA = TupleTools.sortperm(pA[2])
    pA′ = Base.setindex(pA, TupleTools.getindices(pA[2], qA), 2)
    pB′ = Base.setindex(pB, TupleTools.getindices(pB[1], qA), 1)

    qB = TupleTools.sortperm(pB[1])
    pA″ = Base.setindex(pA, TupleTools.getindices(pA[2], qB), 2)
    pB″ = Base.setindex(pB, TupleTools.getindices(pB[1], qB), 1)

    # keep order A en B, check possibilities for cind
    memcost1 = TO.contract_memcost(C, A, pA′, B, pB′, pAB)
    memcost2 = TO.contract_memcost(C, A, pA″, B, pB″, pAB)

    # reverse order A en B, check possibilities for cind
    pAB′ = (
        map(n -> ifelse(n > N₁, n - N₁, n + N₂), pAB[1]),
        map(n -> ifelse(n > N₁, n - N₁, n + N₂), pAB[2]),
    )
    memcost3 = TO.contract_memcost(C, B, reverse(pB′), A, reverse(pA′), pAB′)
    memcost4 = TO.contract_memcost(C, B, reverse(pB″), A, reverse(pA″), pAB′)

    return if min(memcost1, memcost2) <= min(memcost3, memcost4)
        if memcost1 <= memcost2
            return blas_contract!(C, A, pA′, B, pB′, pAB, α, β, backend, allocator)
        else
            return blas_contract!(C, A, pA″, B, pB″, pAB, α, β, backend, allocator)
        end
    else
        if memcost3 <= memcost4
            return blas_contract!(C, B, reverse(pB′), A, reverse(pA′), pAB′, α, β, backend, allocator)
        else
            return blas_contract!(C, B, reverse(pB″), A, reverse(pA″), pAB′, α, β, backend, allocator)
        end
    end
end

function TO.contract_memcost(
        C::AbstractTensorMap,
        A::AbstractTensorMap, pA::Index2Tuple,
        B::AbstractTensorMap, pB::Index2Tuple,
        pAB::Index2Tuple
    )
    ipAB = TO.oindABinC(pAB, pA, pB)
    return dim(A) * (!TO.isblascontractable(A, pA) || eltype(A) !== eltype(C)) +
        dim(B) * (!TO.isblascontractable(B, pB) || eltype(B) !== eltype(C)) +
        dim(C) * !TO.isblasdestination(C, ipAB)
end

function TO.isblascontractable(A::AbstractTensorMap, pA::Index2Tuple)
    return eltype(A) <: LinearAlgebra.BlasFloat && has_shared_permute(A, pA)
end
function TO.isblasdestination(A::AbstractTensorMap, ipAB::Index2Tuple)
    return eltype(A) <: LinearAlgebra.BlasFloat && has_shared_permute(A, ipAB)
end

function blas_contract!(
        C::AbstractTensorMap,
        A::AbstractTensorMap, pA::Index2Tuple,
        B::AbstractTensorMap, pB::Index2Tuple,
        pAB::Index2Tuple, α, β,
        backend, allocator
    )
    bstyle = BraidingStyle(sectortype(C))
    bstyle isa SymmetricBraiding ||
        throw(SectorMismatch("only tensors with symmetric braiding rules can be contracted; try `@planar` instead"))
    TC = eltype(C)

    if !(bstyle isa Fermionic) && FusionStyle(sectortype(C)) isa UniqueFusion && C isa TensorMap && A isa TensorMap && B isa TensorMap
        transformer = contractiontransformer(space(C), space(A), pA, space(B), pB, pAB)
        _contract_abelian_kernel!(C, A, pA, B, pB, pAB, transformer, α, β, backend, allocator)
        return C
    end

    # check which tensors have to be permuted/copied
    copyA = !(TO.isblascontractable(A, pA) && eltype(A) === TC)
    copyB = !(TO.isblascontractable(B, pB) && eltype(B) === TC)

    if bstyle isa Fermionic && any(isdual ∘ Base.Fix1(space, B), pB[1])
        # twist smallest object if neither or both already have to be permuted
        # otherwise twist the one that already is copied
        if copyA ⊻ copyB
            twistA = dim(A) < dim(B)
        else
            twistA = copyA
        end
        twistB = !twistA
        copyA |= twistA
        copyB |= twistB
    else
        twistA = false
        twistB = false
    end

    # Bring A in the correct form for BLAS contraction
    if copyA
        Anew = TO.tensoralloc_add(TC, A, pA, false, Val(true), allocator)
        Anew = TO.tensoradd!(Anew, A, pA, false, One(), Zero(), backend, allocator)
        twistA && twist!(Anew, filter(!isdual ∘ Base.Fix1(space, Anew), domainind(Anew)))
    else
        Anew = permute(A, pA)
    end
    pAnew = (codomainind(Anew), domainind(Anew))

    # Bring B in the correct form for BLAS contraction
    if copyB
        Bnew = TO.tensoralloc_add(TC, B, pB, false, Val(true), allocator)
        Bnew = TO.tensoradd!(Bnew, B, pB, false, One(), Zero(), backend, allocator)
        twistB && twist!(Bnew, filter(isdual ∘ Base.Fix1(space, Bnew), codomainind(Bnew)))
    else
        Bnew = permute(B, pB)
    end
    pBnew = (codomainind(Bnew), domainind(Bnew))

    # Bring C in the correct form for BLAS contraction
    ipAB = TO.oindABinC(pAB, pAnew, pBnew)
    copyC = !TO.isblasdestination(C, ipAB)

    if copyC
        Cnew = TO.tensoralloc_add(TC, C, ipAB, false, Val(true), allocator)
        mul!(Cnew, Anew, Bnew)
        TO.tensoradd!(C, Cnew, pAB, false, α, β, backend, allocator)
        TO.tensorfree!(Cnew, allocator)
    else
        Cnew = permute(C, ipAB)
        mul!(Cnew, Anew, Bnew, α, β)
    end

    copyA && TO.tensorfree!(Anew, allocator)
    copyB && TO.tensorfree!(Bnew, allocator)

    return C
end

# Custom backend
# --------------
function _contract_abelian_kernel!(
        _C, _A, pA, _B, pB, pAB, transformer, α, β, backend, allocator
    )
    isempty(transformer.data) && return nothing

    A = _A.data
    B = _B.data
    C = _C.data

    # preallocate buffers
    A′ = needsbuffer(transformer, 1) ? similar(A) : A
    B′ = needsbuffer(transformer, 2) ? similar(B) : B
    C′ = needsbuffer(transformer, 3) ? similar(C) : C

    if use_threaded_transform(_C, transformer)
        _contract_abelian_kernel_threaded!((C, C′), (A, A′), pA, (B, B′), pB, pAB, transformer, α, β, backend, allocator)
    else
        _contract_abelian_kernel_nonthreaded!((C, C′), (A, A′), pA, (B, B′), pB, pAB, transformer, α, β, backend, allocator)
    end
    return nothing
end

function _contract_abelian_kernel_nonthreaded!(
        C, A, pA, B, pB, pAB, transformer, α, β, backend, allocator
    )
    for subtransformer in transformer.data
        _contract_abelian_single!(C, A, pA, B, pB, pAB, subtransformer, α, β, backend, allocator)
    end
    return nothing
end

function _contract_abelian_kernel_threaded!(
        C, A, pA, B, pB, pAB, transformer, α, β, backend, allocator; ntasks::Int = get_num_transformer_threads()
    )
    nblocks = length(transformer.data)
    counter = Threads.Atomic{Int}(1)
    Threads.@sync for _ in 1:min(ntasks, nblocks)
        Threads.@spawn begin
            while true
                local_counter = Threads.atomic_add!(counter, 1)
                local_counter > nblocks && break
                subtransformer = transformer.data[local_counter]
                _contract_abelian_single!(C, A, pA, B, pB, pAB, subtransformer, α, β, backend, allocator)
            end
        end
    end
    return nothing
end

function _contract_abelian_single!(
        (C, C′), (A, A′), pA, (B, B′), pB, pAB,
        ((bstrA, bstrB, bstrC), transformA, transformB, transformC), α, β, backend, allocator
    )
    A === A′ || _add_abelian_kernel_nonthreaded!(A′, A, pA, transformA, One(), Zero(), backend, allocator)
    B === B′ || _add_abelian_kernel_nonthreaded!(B′, A, pB, transformB, One(), Zero(), backend, allocator)

    bC′ = StridedView(C′, bstrC...)
    bA′ = StridedView(A′, bstrA...)
    bB′ = StridedView(B′, bstrB...)

    if C === C′
        mul!(bC′, bA′, bB′, α, β)
    else
        mul!(bC′, bA′, bB′)
        _add_abelian_kernel_nonthreaded!(C, C′, pAB, transformC, α, β, backend, allocator)
    end

    return nothing
end

# Scalar implementation
#-----------------------
function scalar(t::AbstractTensorMap)
    # TODO: should scalar only work if N₁ == N₂ == 0?
    return dim(codomain(t)) == dim(domain(t)) == 1 ?
        first(blocks(t))[2][1, 1] : throw(DimensionMismatch())
end
