# BASIC MANIPULATIONS:
#----------------------------------------------
# -> rewrite generic fusion tree in basis of fusion trees in standard form
# -> only depend on Fsymbol

"""
    insertat(f::FusionTree{I, N₁}, i::Int, f₂::FusionTree{I, N₂})
    -> <:AbstractDict{<:FusionTree{I, N₁+N₂-1}, <:Number}

Attach a fusion tree `f₂` to the uncoupled leg `i` of the fusion tree `f₁` and bring it
into a linear combination of fusion trees in standard form. This requires that
`f₂.coupled == f₁.uncoupled[i]` and `f₁.isdual[i] == false`.
"""
function insertat(f₁::FusionTree{I}, i::Int, f₂::FusionTree{I, 0}) where {I}
    # this actually removes uncoupled line i, which should be trivial
    (f₁.uncoupled[i] == f₂.coupled && !f₁.isdual[i]) ||
        throw(SectorMismatch("cannot connect $(f₂.uncoupled) to $(f₁.uncoupled[i])"))
    coeff = one(sectorscalartype(I))

    uncoupled = TupleTools.deleteat(f₁.uncoupled, i)
    coupled = f₁.coupled
    isdual = TupleTools.deleteat(f₁.isdual, i)
    if length(uncoupled) <= 2
        inner = ()
    else
        inner = TupleTools.deleteat(f₁.innerlines, max(1, i - 2))
    end
    if length(uncoupled) <= 1
        vertices = ()
    else
        vertices = TupleTools.deleteat(f₁.vertices, max(1, i - 1))
    end
    f = FusionTree(uncoupled, coupled, isdual, inner, vertices)
    return fusiontreedict(I)(f => coeff)
end
function insertat(f₁::FusionTree{I}, i, f₂::FusionTree{I, 1}) where {I}
    # identity operation
    (f₁.uncoupled[i] == f₂.coupled && !f₁.isdual[i]) ||
        throw(SectorMismatch("cannot connect $(f₂.uncoupled) to $(f₁.uncoupled[i])"))
    coeff = one(sectorscalartype(I))
    isdual′ = TupleTools.setindex(f₁.isdual, f₂.isdual[1], i)
    f = FusionTree{I}(f₁.uncoupled, f₁.coupled, isdual′, f₁.innerlines, f₁.vertices)
    return fusiontreedict(I)(f => coeff)
end
function insertat(f₁::FusionTree{I}, i, f₂::FusionTree{I, 2}) where {I}
    # elementary building block,
    (f₁.uncoupled[i] == f₂.coupled && !f₁.isdual[i]) ||
        throw(SectorMismatch("cannot connect $(f₂.uncoupled) to $(f₁.uncoupled[i])"))
    uncoupled = f₁.uncoupled
    coupled = f₁.coupled
    inner = f₁.innerlines
    b, c = f₂.uncoupled
    isdual = f₁.isdual
    isdualb, isdualc = f₂.isdual
    if i == 1
        uncoupled′ = (b, c, tail(uncoupled)...)
        isdual′ = (isdualb, isdualc, tail(isdual)...)
        inner′ = (uncoupled[1], inner...)
        vertices′ = (f₂.vertices..., f₁.vertices...)
        coeff = one(sectorscalartype(I))
        f′ = FusionTree(uncoupled′, coupled, isdual′, inner′, vertices′)
        return fusiontreedict(I)(f′ => coeff)
    end
    uncoupled′ = TupleTools.insertafter(TupleTools.setindex(uncoupled, b, i), i, (c,))
    isdual′ = TupleTools.insertafter(TupleTools.setindex(isdual, isdualb, i), i, (isdualc,))
    inner_extended = (uncoupled[1], inner..., coupled)
    a = inner_extended[i - 1]
    d = inner_extended[i]
    e′ = uncoupled[i]
    if FusionStyle(I) isa MultiplicityFreeFusion
        local newtrees
        for e in a ⊗ b
            coeff = conj(Fsymbol(a, b, c, d, e, e′))
            iszero(coeff) && continue
            inner′ = TupleTools.insertafter(inner, i - 2, (e,))
            f′ = FusionTree(uncoupled′, coupled, isdual′, inner′)
            if @isdefined newtrees
                push!(newtrees, f′ => coeff)
            else
                newtrees = fusiontreedict(I)(f′ => coeff)
            end
        end
        return newtrees
    else
        local newtrees
        κ = f₂.vertices[1]
        λ = f₁.vertices[i - 1]
        for e in a ⊗ b
            inner′ = TupleTools.insertafter(inner, i - 2, (e,))
            Fmat = Fsymbol(a, b, c, d, e, e′)
            for μ in axes(Fmat, 1), ν in axes(Fmat, 2)
                coeff = conj(Fmat[μ, ν, κ, λ])
                iszero(coeff) && continue
                vertices′ = TupleTools.setindex(f₁.vertices, ν, i - 1)
                vertices′ = TupleTools.insertafter(vertices′, i - 2, (μ,))
                f′ = FusionTree(uncoupled′, coupled, isdual′, inner′, vertices′)
                if @isdefined newtrees
                    push!(newtrees, f′ => coeff)
                else
                    newtrees = fusiontreedict(I)(f′ => coeff)
                end
            end
        end
        return newtrees
    end
end
function insertat(f₁::FusionTree{I, N₁}, i, f₂::FusionTree{I, N₂}) where {I, N₁, N₂}
    F = fusiontreetype(I, N₁ + N₂ - 1)
    (f₁.uncoupled[i] == f₂.coupled && !f₁.isdual[i]) ||
        throw(SectorMismatch("cannot connect $(f₂.uncoupled) to $(f₁.uncoupled[i])"))
    T = sectorscalartype(I)
    coeff = one(T)
    if length(f₁) == 1
        return fusiontreedict(I){F, T}(f₂ => coeff)
    end
    if i == 1
        uncoupled = (f₂.uncoupled..., tail(f₁.uncoupled)...)
        isdual = (f₂.isdual..., tail(f₁.isdual)...)
        inner = (f₂.innerlines..., f₂.coupled, f₁.innerlines...)
        vertices = (f₂.vertices..., f₁.vertices...)
        coupled = f₁.coupled
        f′ = FusionTree(uncoupled, coupled, isdual, inner, vertices)
        return fusiontreedict(I){F, T}(f′ => coeff)
    else # recursive definition
        N2 = length(f₂)
        f₂′, f₂′′ = split(f₂, N2 - 1)
        local newtrees::fusiontreedict(I){F, T}
        for (f, coeff) in insertat(f₁, i, f₂′′)
            for (f′, coeff′) in insertat(f, i, f₂′)
                if @isdefined newtrees
                    coeff′′ = coeff * coeff′
                    newtrees[f′] = get(newtrees, f′, zero(coeff′′)) + coeff′′
                else
                    newtrees = fusiontreedict(I){F, T}(f′ => coeff * coeff′)
                end
            end
        end
        return newtrees
    end
end

"""
    split(f::FusionTree{I, N}, M::Int)
    -> (::FusionTree{I, M}, ::FusionTree{I, N-M+1})

Split a fusion tree into two. The first tree has as uncoupled sectors the first `M`
uncoupled sectors of the input tree `f`, whereas its coupled sector corresponds to the
internal sector between uncoupled sectors `M` and `M+1` of the original tree `f`. The
second tree has as first uncoupled sector that same internal sector of `f`, followed by
remaining `N-M` uncoupled sectors of `f`. It couples to the same sector as `f`. This
operation is the inverse of `insertat` in the sense that if
`f₁, f₂ = split(t, M) ⇒ f == insertat(f₂, 1, f₁)`.
"""
@inline function split(f::FusionTree{I, N}, M::Int) where {I, N}
    if M > N || M < 0
        throw(ArgumentError("M should be between 0 and N = $N"))
    elseif M === N
        (f, FusionTree{I}((f.coupled,), f.coupled, (false,), (), ()))
    elseif M === 1
        isdual1 = (f.isdual[1],)
        isdual2 = TupleTools.setindex(f.isdual, false, 1)
        f₁ = FusionTree{I}((f.uncoupled[1],), f.uncoupled[1], isdual1, (), ())
        f₂ = FusionTree{I}(f.uncoupled, f.coupled, isdual2, f.innerlines, f.vertices)
        return f₁, f₂
    elseif M === 0
        u = leftunit(f.uncoupled[1])
        f₁ = FusionTree{I}((), u, (), ())
        uncoupled2 = (u, f.uncoupled...)
        coupled2 = f.coupled
        isdual2 = (false, f.isdual...)
        innerlines2 = N >= 2 ? (f.uncoupled[1], f.innerlines...) : ()
        if FusionStyle(I) isa GenericFusion
            vertices2 = (1, f.vertices...)
            return f₁, FusionTree{I}(uncoupled2, coupled2, isdual2, innerlines2, vertices2)
        else
            return f₁, FusionTree{I}(uncoupled2, coupled2, isdual2, innerlines2)
        end
    else
        uncoupled1 = ntuple(n -> f.uncoupled[n], M)
        isdual1 = ntuple(n -> f.isdual[n], M)
        innerlines1 = ntuple(n -> f.innerlines[n], max(0, M - 2))
        coupled1 = f.innerlines[M - 1]
        vertices1 = ntuple(n -> f.vertices[n], M - 1)

        uncoupled2 = ntuple(N - M + 1) do n
            return n == 1 ? f.innerlines[M - 1] : f.uncoupled[M + n - 1]
        end
        isdual2 = ntuple(N - M + 1) do n
            return n == 1 ? false : f.isdual[M + n - 1]
        end
        innerlines2 = ntuple(n -> f.innerlines[M - 1 + n], N - M - 1)
        coupled2 = f.coupled
        vertices2 = ntuple(n -> f.vertices[M - 1 + n], N - M)

        f₁ = FusionTree{I}(uncoupled1, coupled1, isdual1, innerlines1, vertices1)
        f₂ = FusionTree{I}(uncoupled2, coupled2, isdual2, innerlines2, vertices2)
        return f₁, f₂
    end
end

"""
    merge(f₁::FusionTree{I, N₁}, f₂::FusionTree{I, N₂}, c::I, μ = 1)
    -> <:AbstractDict{<:FusionTree{I, N₁+N₂}, <:Number}

Merge two fusion trees together to a linear combination of fusion trees whose uncoupled
sectors are those of `f₁` followed by those of `f₂`, and where the two coupled sectors of
`f₁` and `f₂` are further fused to `c`. In case of
`FusionStyle(I) == GenericFusion()`, also a degeneracy label `μ` for the fusion of
the coupled sectors of `f₁` and `f₂` to `c` needs to be specified.
"""
function merge(f₁::FusionTree{I, N₁}, f₂::FusionTree{I, N₂}, c::I) where {I, N₁, N₂}
    if FusionStyle(I) isa GenericFusion
        throw(ArgumentError("vertex label for merging required"))
    end
    return merge(f₁, f₂, c, 1)
end
function merge(f₁::FusionTree{I, N₁}, f₂::FusionTree{I, N₂}, c::I, μ) where {I, N₁, N₂}
    if !(c in f₁.coupled ⊗ f₂.coupled)
        throw(SectorMismatch("cannot fuse sectors $(f₁.coupled) and $(f₂.coupled) to $c"))
    end
    if μ > Nsymbol(f₁.coupled, f₂.coupled, c)
        throw(ArgumentError("invalid fusion vertex label $μ"))
    end
    f₀ = FusionTree{I}((f₁.coupled, f₂.coupled), c, (false, false), (), (μ,))
    f, coeff = first(insertat(f₀, 1, f₁)) # takes fast path, single output
    @assert coeff == one(coeff)
    return insertat(f, N₁ + 1, f₂)
end
function merge(f₁::FusionTree{I, 0}, f₂::FusionTree{I, 0}, c::I, μ) where {I}
    Nsymbol(f₁.coupled, f₂.coupled, c) == μ == 1 ||
        throw(SectorMismatch("cannot fuse sectors $(f₁.coupled) and $(f₂.coupled) to $c"))
    return fusiontreedict(I)(f₁ => Fsymbol(c, c, c, c, c, c)[1, 1, 1, 1])
end

# flip a duality flag of a fusion tree
function flip((f₁, f₂)::FusionTreePair{I, N₁, N₂}, i::Int; inv::Bool = false) where {I, N₁, N₂}
    @assert 0 < i ≤ N₁ + N₂
    if i ≤ N₁
        a = f₁.uncoupled[i]
        χₐ = frobenius_schur_phase(a)
        θₐ = twist(a)
        if !inv
            factor = f₁.isdual[i] ? χₐ * θₐ : one(θₐ)
        else
            factor = f₁.isdual[i] ? one(θₐ) : conj(χₐ * θₐ)
        end
        isdual′ = TupleTools.setindex(f₁.isdual, !f₁.isdual[i], i)
        f₁′ = FusionTree{I}(f₁.uncoupled, f₁.coupled, isdual′, f₁.innerlines, f₁.vertices)
        return SingletonDict((f₁′, f₂) => factor)
    else
        i -= N₁
        a = f₂.uncoupled[i]
        χₐ = frobenius_schur_phase(a)
        θₐ = twist(a)
        if !inv
            factor = f₂.isdual[i] ? conj(χₐ) * one(θₐ) : θₐ
        else
            factor = f₂.isdual[i] ? conj(θₐ) : χₐ * one(θₐ)
        end
        isdual′ = TupleTools.setindex(f₂.isdual, !f₂.isdual[i], i)
        f₂′ = FusionTree{I}(f₂.uncoupled, f₂.coupled, isdual′, f₂.innerlines, f₂.vertices)
        return SingletonDict((f₁, f₂′) => factor)
    end
end
function flip((f₁, f₂)::FusionTreePair{I, N₁, N₂}, ind; inv::Bool = false) where {I, N₁, N₂}
    f₁′, f₂′ = f₁, f₂
    factor = one(sectorscalartype(I))
    for i in ind
        (f₁′, f₂′), s = only(flip((f₁′, f₂′), i; inv))
        factor *= s
    end
    return SingletonDict((f₁′, f₂′) => factor)
end
