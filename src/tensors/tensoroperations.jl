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
        A::AbstractTensorMap, (oindA, cindA)::Index2Tuple,
        B::AbstractTensorMap, (cindB, oindB)::Index2Tuple,
        (p₁, p₂)::Index2Tuple,
        α::Number, β::Number,
        backend, allocator
    )
    length(cindA) == length(cindB) ||
        throw(IndexError("number of contracted indices does not match"))
    N₁, N₂ = length(oindA), length(oindB)

    # find optimal contraction scheme
    hsp = has_shared_permute
    ipAB = TupleTools.invperm((p₁..., p₂...))
    oindAinC = TupleTools.getindices(ipAB, ntuple(n -> n, N₁))
    oindBinC = TupleTools.getindices(ipAB, ntuple(n -> n + N₁, N₂))

    qA = TupleTools.sortperm(cindA)
    cindA′ = TupleTools.getindices(cindA, qA)
    cindB′ = TupleTools.getindices(cindB, qA)

    qB = TupleTools.sortperm(cindB)
    cindA′′ = TupleTools.getindices(cindA, qB)
    cindB′′ = TupleTools.getindices(cindB, qB)

    dA, dB, dC = dim(A), dim(B), dim(C)

    # keep order A en B, check possibilities for cind
    memcost1 = memcost2 = dC * (!hsp(C, (oindAinC, oindBinC)))
    memcost1 += dA * (!hsp(A, (oindA, cindA′))) + dB * (!hsp(B, (cindB′, oindB)))
    memcost2 += dA * (!hsp(A, (oindA, cindA′′))) + dB * (!hsp(B, (cindB′′, oindB)))

    # reverse order A en B, check possibilities for cind
    memcost3 = memcost4 = dC * (!hsp(C, (oindBinC, oindAinC)))
    memcost3 += dB * (!hsp(B, (oindB, cindB′))) + dA * (!hsp(A, (cindA′, oindA)))
    memcost4 += dB * (!hsp(B, (oindB, cindB′′))) + dA * (!hsp(A, (cindA′′, oindA)))

    return if min(memcost1, memcost2) <= min(memcost3, memcost4)
        if memcost1 <= memcost2
            return _contract!(α, A, B, β, C, oindA, cindA′, oindB, cindB′, p₁, p₂)
        else
            return _contract!(α, A, B, β, C, oindA, cindA′′, oindB, cindB′′, p₁, p₂)
        end
    else
        p1′ = map(n -> ifelse(n > N₁, n - N₁, n + N₂), p₁)
        p2′ = map(n -> ifelse(n > N₁, n - N₁, n + N₂), p₂)
        if memcost3 <= memcost4
            return _contract!(α, B, A, β, C, oindB, cindB′, oindA, cindA′, p1′, p2′)
        else
            return _contract!(α, B, A, β, C, oindB, cindB′′, oindA, cindA′′, p1′, p2′)
        end
    end
end

# TODO: also transform _contract! into new interface, and add backend support
function _contract!(
        α, A::AbstractTensorMap, B::AbstractTensorMap,
        β, C::AbstractTensorMap,
        oindA::IndexTuple, cindA::IndexTuple,
        oindB::IndexTuple, cindB::IndexTuple,
        p₁::IndexTuple, p₂::IndexTuple
    )
    if !(BraidingStyle(sectortype(C)) isa SymmetricBraiding)
        throw(SectorMismatch("only tensors with symmetric braiding rules can be contracted; try `@planar` instead"))
    end
    N₁, N₂ = length(oindA), length(oindB)
    copyA = false
    if BraidingStyle(sectortype(A)) isa Fermionic
        for i in cindA
            if !isdual(space(A, i))
                copyA = true
            end
        end
    end
    A′ = permute(A, (oindA, cindA); copy = copyA)
    B′ = permute(B, (cindB, oindB))
    if BraidingStyle(sectortype(A)) isa Fermionic
        for i in domainind(A′)
            if !isdual(space(A′, i))
                A′ = twist!(A′, i)
            end
        end
        # A′ = twist!(A′, filter(i -> !isdual(space(A′, i)), domainind(A′)))
        # commented version leads to boxing of `A′` and type instabilities in the result
    end
    ipAB = TupleTools.invperm((p₁..., p₂...))
    oindAinC = TupleTools.getindices(ipAB, ntuple(n -> n, N₁))
    oindBinC = TupleTools.getindices(ipAB, ntuple(n -> n + N₁, N₂))
    if has_shared_permute(C, (oindAinC, oindBinC))
        C′ = permute(C, (oindAinC, oindBinC))
        mul!(C′, A′, B′, α, β)
    else
        C′ = A′ * B′
        add_permute!(C, C′, (p₁, p₂), α, β)
    end
    return C
end

# Scalar implementation
#-----------------------
function scalar(t::AbstractTensorMap{T, S, 0, 0}) where {T, S}
    Bs = collect(blocks(t))
    inds = findall(!iszero ∘ last, Bs)
    isempty(inds) && return zero(scalartype(t))
    return only(last(Bs[only(inds)]))
end
