fusiontreedict(I) = FusionStyle(I) isa UniqueFusion ? SingletonDict : FusionTreeDict

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

# ELEMENTARY DUALITY MANIPULATIONS: A- and B-moves
#---------------------------------------------------------
# -> elementary manipulations that depend on the duality (rigidity) and pivotal structure
# -> planar manipulations that do not require braiding, everything is in Fsymbol (A/Bsymbol)
# -> B-move (bendleft, bendright) is simple in standard basis
# -> A-move (foldleft, foldright) is complicated, needs to be reexpressed in standard form

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
            factor = f₁.isdual[i] ? one(θₐ) : χₐ * conj(θₐ)
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
            factor = f₂.isdual[i] ? χₐ * one(θₐ) : θₐ
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

# change to N₁ - 1, N₂ + 1
function bendright((f₁, f₂)::FusionTreePair{I, N₁, N₂}) where {I, N₁, N₂}
    # map final splitting vertex (a, b)<-c to fusion vertex a<-(c, dual(b))
    @assert N₁ > 0
    c = f₁.coupled
    a = N₁ == 1 ? leftunit(f₁.uncoupled[1]) :
        (N₁ == 2 ? f₁.uncoupled[1] : f₁.innerlines[end])
    b = f₁.uncoupled[N₁]

    uncoupled1 = TupleTools.front(f₁.uncoupled)
    isdual1 = TupleTools.front(f₁.isdual)
    inner1 = N₁ > 2 ? TupleTools.front(f₁.innerlines) : ()
    vertices1 = N₁ > 1 ? TupleTools.front(f₁.vertices) : ()
    f₁′ = FusionTree(uncoupled1, a, isdual1, inner1, vertices1)

    uncoupled2 = (f₂.uncoupled..., dual(b))
    isdual2 = (f₂.isdual..., !(f₁.isdual[N₁]))
    inner2 = N₂ > 1 ? (f₂.innerlines..., c) : ()

    coeff₀ = sqrtdim(c) * invsqrtdim(a)
    if f₁.isdual[N₁]
        coeff₀ *= conj(frobenius_schur_phase(dual(b)))
    end
    if FusionStyle(I) isa MultiplicityFreeFusion
        coeff = coeff₀ * Bsymbol(a, b, c)
        vertices2 = N₂ > 0 ? (f₂.vertices..., 1) : ()
        f₂′ = FusionTree(uncoupled2, a, isdual2, inner2, vertices2)
        return SingletonDict((f₁′, f₂′) => coeff)
    else
        local newtrees
        Bmat = Bsymbol(a, b, c)
        μ = N₁ > 1 ? f₁.vertices[end] : 1
        for ν in axes(Bmat, 2)
            coeff = coeff₀ * Bmat[μ, ν]
            iszero(coeff) && continue
            vertices2 = N₂ > 0 ? (f₂.vertices..., ν) : ()
            f₂′ = FusionTree(uncoupled2, a, isdual2, inner2, vertices2)
            if @isdefined newtrees
                push!(newtrees, (f₁′, f₂′) => coeff)
            else
                newtrees = FusionTreeDict((f₁′, f₂′) => coeff)
            end
        end
        return newtrees
    end
end
# change to N₁ + 1, N₂ - 1
function bendleft((f₁, f₂)::FusionTreePair{I}) where {I}
    # map final fusion vertex c<-(a, b) to splitting vertex (c, dual(b))<-a
    return fusiontreedict(I)(
        (f₁′, f₂′) => conj(coeff)
            for ((f₂′, f₁′), coeff) in bendright((f₂, f₁))
    )
end

# change to N₁ - 1, N₂ + 1
function foldright((f₁, f₂)::FusionTreePair{I, N₁, N₂}) where {I, N₁, N₂}
    # map first splitting vertex (a, b)<-c to fusion vertex b<-(dual(a), c)
    @assert N₁ > 0
    a = f₁.uncoupled[1]
    isduala = f₁.isdual[1]
    factor = sqrtdim(a)
    if !isduala
        factor *= conj(frobenius_schur_phase(a))
    end
    c1 = dual(a)
    c2 = f₁.coupled
    uncoupled = Base.tail(f₁.uncoupled)
    isdual = Base.tail(f₁.isdual)
    if FusionStyle(I) isa UniqueFusion
        c = first(c1 ⊗ c2)
        fl = FusionTree{I}(Base.tail(f₁.uncoupled), c, Base.tail(f₁.isdual))
        fr = FusionTree{I}((c1, f₂.uncoupled...), c, (!isduala, f₂.isdual...))
        return fusiontreedict(I)((fl, fr) => factor)
    else
        local newtrees
        if N₁ == 1
            cset = (leftunit(c1),) # or rightunit(a)
        elseif N₁ == 2
            cset = (f₁.uncoupled[2],)
        else
            cset = ⊗(Base.tail(f₁.uncoupled)...)
        end
        for c in c1 ⊗ c2
            c ∈ cset || continue
            for μ in 1:Nsymbol(c1, c2, c)
                fc = FusionTree((c1, c2), c, (!isduala, false), (), (μ,))
                fr_coeffs = insertat(fc, 2, f₂)
                for (fl′, coeff1) in insertat(fc, 2, f₁)
                    N₁ > 1 && !isunit(fl′.innerlines[1]) && continue
                    coupled = fl′.coupled
                    uncoupled = Base.tail(Base.tail(fl′.uncoupled))
                    isdual = Base.tail(Base.tail(fl′.isdual))
                    inner = N₁ <= 3 ? () : Base.tail(Base.tail(fl′.innerlines))
                    vertices = N₁ <= 2 ? () : Base.tail(Base.tail(fl′.vertices))
                    fl = FusionTree{I}(uncoupled, coupled, isdual, inner, vertices)
                    for (fr, coeff2) in fr_coeffs
                        coeff = factor * coeff1 * conj(coeff2)
                        if (@isdefined newtrees)
                            newtrees[(fl, fr)] = get(newtrees, (fl, fr), zero(coeff)) +
                                coeff
                        else
                            newtrees = fusiontreedict(I)((fl, fr) => coeff)
                        end
                    end
                end
            end
        end
        return newtrees
    end
end
# change to N₁ + 1, N₂ - 1
function foldleft((f₁, f₂)::FusionTreePair{I}) where {I}
    # map first fusion vertex c<-(a, b) to splitting vertex (dual(a), c)<-b
    return fusiontreedict(I)(
        (f₁′, f₂′) => conj(coeff)
            for ((f₂′, f₁′), coeff) in foldright((f₂, f₁))
    )
end

# COMPOSITE DUALITY MANIPULATIONS PART 1: Repartition and transpose
#-------------------------------------------------------------------
# -> composite manipulations that depend on the duality (rigidity) and pivotal structure
# -> planar manipulations that do not require braiding, everything is in Fsymbol (A/Bsymbol)
# -> transpose expressed as cyclic permutation
# one-argument version: check whether `p` is a cyclic permutation (of `1:length(p)`)
function iscyclicpermutation(p)
    N = length(p)
    @inbounds for i in 1:N
        p[mod1(i + 1, N)] == mod1(p[i] + 1, N) || return false
    end
    return true
end
# two-argument version: check whether `v1` is a cyclic permutation of `v2`
function iscyclicpermutation(v1, v2)
    length(v1) == length(v2) || return false
    return iscyclicpermutation(indexin(v1, v2))
end

# clockwise cyclic permutation while preserving (N₁, N₂): foldright & bendleft
function cycleclockwise((f₁, f₂)::FusionTreePair{I}) where {I}
    local newtrees
    if length(f₁) > 0
        for ((f1a, f2a), coeffa) in foldright((f₁, f₂))
            for ((f1b, f2b), coeffb) in bendleft((f1a, f2a))
                coeff = coeffa * coeffb
                if (@isdefined newtrees)
                    newtrees[(f1b, f2b)] = get(newtrees, (f1b, f2b), zero(coeff)) + coeff
                else
                    newtrees = fusiontreedict(I)((f1b, f2b) => coeff)
                end
            end
        end
    else
        for ((f1a, f2a), coeffa) in bendleft((f₁, f₂))
            for ((f1b, f2b), coeffb) in foldright((f1a, f2a))
                coeff = coeffa * coeffb
                if (@isdefined newtrees)
                    newtrees[(f1b, f2b)] = get(newtrees, (f1b, f2b), zero(coeff)) + coeff
                else
                    newtrees = fusiontreedict(I)((f1b, f2b) => coeff)
                end
            end
        end
    end
    return newtrees
end

# anticlockwise cyclic permutation while preserving (N₁, N₂): foldleft & bendright
function cycleanticlockwise((f₁, f₂)::FusionTreePair{I}) where {I}
    local newtrees
    if length(f₂) > 0
        for ((f1a, f2a), coeffa) in foldleft((f₁, f₂))
            for ((f1b, f2b), coeffb) in bendright((f1a, f2a))
                coeff = coeffa * coeffb
                if (@isdefined newtrees)
                    newtrees[(f1b, f2b)] = get(newtrees, (f1b, f2b), zero(coeff)) + coeff
                else
                    newtrees = fusiontreedict(I)((f1b, f2b) => coeff)
                end
            end
        end
    else
        for ((f1a, f2a), coeffa) in bendright((f₁, f₂))
            for ((f1b, f2b), coeffb) in foldleft((f1a, f2a))
                coeff = coeffa * coeffb
                if (@isdefined newtrees)
                    newtrees[(f1b, f2b)] = get(newtrees, (f1b, f2b), zero(coeff)) + coeff
                else
                    newtrees = fusiontreedict(I)((f1b, f2b) => coeff)
                end
            end
        end
    end
    return newtrees
end

# repartition double fusion tree
"""
    repartition((f₁, f₂)::FusionTreePair{I, N₁, N₂}, N::Int) where {I, N₁, N₂}
        -> <:AbstractDict{<:FusionTreePair{I, N, N₁+N₂-N}}, <:Number}

Input is a double fusion tree that describes the fusion of a set of incoming uncoupled
sectors to a set of outgoing uncoupled sectors, represented using the individual trees of
outgoing (`f₁`) and incoming sectors (`f₂`) respectively (with identical coupled sector
`f₁.coupled == f₂.coupled`). Computes new trees and corresponding coefficients obtained from
repartitioning the tree by bending incoming to outgoing sectors (or vice versa) in order to
have `N` outgoing sectors.
"""
@inline function repartition((f₁, f₂)::FusionTreePair, N::Int)
    f₁.coupled == f₂.coupled || throw(SectorMismatch())
    @assert 0 <= N <= length(f₁) + length(f₂)
    return _recursive_repartition((f₁, f₂), Val(N))
end

function _recursive_repartition((f₁, f₂)::FusionTreePair{I, N₁, N₂}, ::Val{N}) where {I, N₁, N₂, N}
    # recursive definition is only way to get correct number of loops for
    # GenericFusion, but is too complex for type inference to handle, so we
    # precompute the parameters of the return type
    F₁ = fusiontreetype(I, N)
    F₂ = fusiontreetype(I, N₁ + N₂ - N)
    FF = Tuple{F₁, F₂}
    T = sectorscalartype(I)
    coeff = one(T)
    if N == N₁
        return fusiontreedict(I){Tuple{F₁, F₂}, T}((f₁, f₂) => coeff)
    else
        local newtrees::fusiontreedict(I){Tuple{F₁, F₂}, T}
        for ((f₁′, f₂′), coeff1) in (N < N₁ ? bendright((f₁, f₂)) : bendleft((f₁, f₂)))
            for ((f₁′′, f₂′′), coeff2) in _recursive_repartition((f₁′, f₂′), Val(N))
                if (@isdefined newtrees)
                    push!(newtrees, (f₁′′, f₂′′) => coeff1 * coeff2)
                else
                    newtrees = fusiontreedict(I){FF, T}((f₁′′, f₂′′) => coeff1 * coeff2)
                end
            end
        end
        return newtrees
    end
end

"""
    transpose((f₁, f₂)::FusionTreePair{I}, p::(Index2Tuple{N₁, N₂}) where {I, N₁, N₂}
        -> <:AbstractDict{<:FusionTreePair{I, N₁, N₂}}, <:Number}

Input is a double fusion tree that describes the fusion of a set of incoming uncoupled
sectors to a set of outgoing uncoupled sectors, represented using the individual trees of
outgoing (`t1`) and incoming sectors (`t2`) respectively (with identical coupled sector
`t1.coupled == t2.coupled`). Computes new trees and corresponding coefficients obtained from
repartitioning and permuting the tree such that sectors `p1` become outgoing and sectors
`p2` become incoming.
"""
function Base.transpose((f₁, f₂)::FusionTreePair{I}, p::Index2Tuple{N₁, N₂}) where {I, N₁, N₂}
    N = N₁ + N₂
    @assert length(f₁) + length(f₂) == N
    p′ = linearizepermutation(p..., length(f₁), length(f₂))
    @assert iscyclicpermutation(p′)
    return fstranspose(((f₁, f₂), p))
end

const FSTransposeKey{I, N₁, N₂} = Tuple{<:FusionTreePair{I}, Index2Tuple{N₁, N₂}}

function _fsdicttype(I, N₁, N₂)
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)
    T = sectorscalartype(I)
    return fusiontreedict(I){Tuple{F₁, F₂}, T}
end

@cached function fstranspose(key::FSTransposeKey{I, N₁, N₂})::_fsdicttype(I, N₁, N₂) where {I, N₁, N₂}
    (f₁, f₂), (p1, p2) = key
    N = N₁ + N₂
    p = linearizepermutation(p1, p2, length(f₁), length(f₂))
    newtrees = repartition((f₁, f₂), N₁)
    length(p) == 0 && return newtrees
    i1 = findfirst(==(1), p)
    @assert i1 !== nothing
    i1 == 1 && return newtrees
    Nhalf = N >> 1
    while 1 < i1 <= Nhalf
        local newtrees′
        for ((f1a, f2a), coeffa) in newtrees
            for ((f1b, f2b), coeffb) in cycleanticlockwise((f1a, f2a))
                coeff = coeffa * coeffb
                if (@isdefined newtrees′)
                    newtrees′[(f1b, f2b)] = get(newtrees′, (f1b, f2b), zero(coeff)) + coeff
                else
                    newtrees′ = fusiontreedict(I)((f1b, f2b) => coeff)
                end
            end
        end
        newtrees = newtrees′
        i1 -= 1
    end
    while Nhalf < i1
        local newtrees′
        for ((f1a, f2a), coeffa) in newtrees
            for ((f1b, f2b), coeffb) in cycleclockwise((f1a, f2a))
                coeff = coeffa * coeffb
                if (@isdefined newtrees′)
                    newtrees′[(f1b, f2b)] = get(newtrees′, (f1b, f2b), zero(coeff)) + coeff
                else
                    newtrees′ = fusiontreedict(I)((f1b, f2b) => coeff)
                end
            end
        end
        newtrees = newtrees′
        i1 = mod1(i1 + 1, N)
    end
    return newtrees
end

function CacheStyle(::typeof(fstranspose), k::FSTransposeKey{I}) where {I}
    if FusionStyle(I) isa UniqueFusion
        return NoCache()
    else
        return GlobalLRUCache()
    end
end

# COMPOSITE DUALITY MANIPULATIONS PART 2: Planar traces
#-------------------------------------------------------------------
# -> composite manipulations that depend on the duality (rigidity) and pivotal structure
# -> planar manipulations that do not require braiding, everything is in Fsymbol (A/Bsymbol)

function planar_trace(
        (f₁, f₂)::FusionTreePair{I}, (p1, p2)::Index2Tuple{N₁, N₂}, (q1, q2)::Index2Tuple{N₃, N₃}
    ) where {I, N₁, N₂, N₃}
    N = N₁ + N₂ + 2N₃
    @assert length(f₁) + length(f₂) == N
    if N₃ == 0
        return transpose((f₁, f₂), (p1, p2))
    end

    linearindex = (
        ntuple(identity, Val(length(f₁)))...,
        reverse(length(f₁) .+ ntuple(identity, Val(length(f₂))))...,
    )

    q1′ = TupleTools.getindices(linearindex, q1)
    q2′ = TupleTools.getindices(linearindex, q2)
    p1′, p2′ = let q′ = (q1′..., q2′...)
        (
            map(l -> l - count(l .> q′), TupleTools.getindices(linearindex, p1)),
            map(l -> l - count(l .> q′), TupleTools.getindices(linearindex, p2)),
        )
    end

    T = sectorscalartype(I)
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)
    newtrees = FusionTreeDict{Tuple{F₁, F₂}, T}()
    for ((f₁′, f₂′), coeff′) in repartition((f₁, f₂), N)
        for (f₁′′, coeff′′) in planar_trace(f₁′, (q1′, q2′))
            for (f12′′′, coeff′′′) in transpose((f₁′′, f₂′), (p1′, p2′))
                coeff = coeff′ * coeff′′ * coeff′′′
                if !iszero(coeff)
                    newtrees[f12′′′] = get(newtrees, f12′′′, zero(coeff)) + coeff
                end
            end
        end
    end
    return newtrees
end

"""
    planar_trace(f::FusionTree{I,N}, (q1, q2)::Index2Tuple{N₃,N₃}) where {I,N,N₃}
        -> <:AbstractDict{FusionTree{I,N-2*N₃}, <:Number}

Perform a planar trace of the uncoupled indices of the fusion tree `f` at `q1` with those at
`q2`, where `q1[i]` is connected to `q2[i]` for all `i`. The result is returned as a dictionary
of output trees and corresponding coefficients.
"""
function planar_trace(f::FusionTree{I, N}, (q1, q2)::Index2Tuple{N₃, N₃}) where {I, N, N₃}
    T = sectorscalartype(I)
    F = fusiontreetype(I, N - 2 * N₃)
    newtrees = FusionTreeDict{F, T}()
    N₃ === 0 && return push!(newtrees, f => one(T))

    for (i, j) in zip(q1, q2)
        (f.uncoupled[i] == dual(f.uncoupled[j]) && f.isdual[i] != f.isdual[j]) ||
            return newtrees
    end
    k = 1
    local i, j
    while k <= N₃
        if mod1(q1[k] + 1, N) == q2[k]
            i = q1[k]
            j = q2[k]
            break
        elseif mod1(q2[k] + 1, N) == q1[k]
            i = q2[k]
            j = q1[k]
            break
        else
            k += 1
        end
    end
    k > N₃ && throw(ArgumentError("Not a planar trace"))

    q1′ = let i = i, j = j
        map(l -> (l - (l > i) - (l > j)), TupleTools.deleteat(q1, k))
    end
    q2′ = let i = i, j = j
        map(l -> (l - (l > i) - (l > j)), TupleTools.deleteat(q2, k))
    end
    for (f′, coeff′) in elementary_trace(f, i)
        for (f′′, coeff′′) in planar_trace(f′, (q1′, q2′))
            coeff = coeff′ * coeff′′
            if !iszero(coeff)
                newtrees[f′′] = get(newtrees, f′′, zero(coeff)) + coeff
            end
        end
    end
    return newtrees
end

# trace two neighbouring indices of a single fusion tree
"""
    elementary_trace(f::FusionTree{I,N}, i) where {I,N} -> <:AbstractDict{FusionTree{I,N-2}, <:Number}

Perform an elementary trace of neighbouring uncoupled indices `i` and
`i+1` on a fusion tree `f`, and returns the result as a dictionary of output trees and
corresponding coefficients.
"""
function elementary_trace(f::FusionTree{I, N}, i) where {I, N}
    (N > 1 && 1 <= i <= N) ||
        throw(ArgumentError("Cannot trace outputs i=$i and i+1 out of only $N outputs"))
    i < N || isunit(f.coupled) ||
        throw(ArgumentError("Cannot trace outputs i=$N and 1 of fusion tree that couples to non-trivial sector"))

    T = sectorscalartype(I)
    F = fusiontreetype(I, N - 2)
    newtrees = FusionTreeDict{F, T}()

    j = mod1(i + 1, N)
    b = f.uncoupled[i]
    b′ = f.uncoupled[j]
    # if trace is zero, return empty dict
    (b == dual(b′) && f.isdual[i] != f.isdual[j]) || return newtrees
    if i < N
        inner_extended = (leftunit(f.uncoupled[1]), f.uncoupled[1], f.innerlines..., f.coupled)
        a = inner_extended[i]
        d = inner_extended[i + 2]
        a == d || return newtrees
        uncoupled′ = TupleTools.deleteat(TupleTools.deleteat(f.uncoupled, i + 1), i)
        isdual′ = TupleTools.deleteat(TupleTools.deleteat(f.isdual, i + 1), i)
        coupled′ = f.coupled
        if N <= 4
            inner′ = ()
        else
            inner′ = i <= 2 ? Base.tail(Base.tail(f.innerlines)) :
                TupleTools.deleteat(TupleTools.deleteat(f.innerlines, i - 1), i - 2)
        end
        if N <= 3
            vertices′ = ()
        else
            vertices′ = i <= 2 ? Base.tail(Base.tail(f.vertices)) :
                TupleTools.deleteat(TupleTools.deleteat(f.vertices, i), i - 1)
        end
        f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner′, vertices′)
        coeff = sqrtdim(b)
        if i > 1
            c = f.innerlines[i - 1]
            if FusionStyle(I) isa MultiplicityFreeFusion
                coeff *= Fsymbol(a, b, dual(b), a, c, rightunit(a))
            else
                μ = f.vertices[i - 1]
                ν = f.vertices[i]
                coeff *= Fsymbol(a, b, dual(b), a, c, rightunit(a))[μ, ν, 1, 1]
            end
        end
        if f.isdual[i]
            coeff *= frobenius_schur_phase(b)
        end
        push!(newtrees, f′ => coeff)
        return newtrees
    else # i == N
        unit = leftunit(b)
        if N == 2
            f′ = FusionTree{I}((), unit, (), (), ())
            coeff = sqrtdim(b)
            if !(f.isdual[N])
                coeff *= conj(frobenius_schur_phase(b))
            end
            push!(newtrees, f′ => coeff)
            return newtrees
        end
        uncoupled_ = TupleTools.front(f.uncoupled)
        inner_ = TupleTools.front(f.innerlines)
        coupled_ = f.innerlines[end]
        isdual_ = TupleTools.front(f.isdual)
        vertices_ = TupleTools.front(f.vertices)
        f_ = FusionTree(uncoupled_, coupled_, isdual_, inner_, vertices_)
        fs = FusionTree((b,), b, (!f.isdual[1],), (), ())
        for (f_′, coeff) in merge(fs, f_, unit, 1)
            f_′.innerlines[1] == unit || continue
            uncoupled′ = Base.tail(Base.tail(f_′.uncoupled))
            isdual′ = Base.tail(Base.tail(f_′.isdual))
            inner′ = N <= 4 ? () : Base.tail(Base.tail(f_′.innerlines))
            vertices′ = N <= 3 ? () : Base.tail(Base.tail(f_′.vertices))
            f′ = FusionTree(uncoupled′, unit, isdual′, inner′, vertices′)
            coeff *= sqrtdim(b)
            if !(f.isdual[N])
                coeff *= conj(frobenius_schur_phase(b))
            end
            newtrees[f′] = get(newtrees, f′, zero(coeff)) + coeff
        end
        return newtrees
    end
end

# BRAIDING MANIPULATIONS:
#-----------------------------------------------
# -> manipulations that depend on a braiding
# -> requires both Fsymbol and Rsymbol
"""
    artin_braid(f::FusionTree, i; inv::Bool = false) -> <:AbstractDict{typeof(f), <:Number}

Perform an elementary braid (Artin generator) of neighbouring uncoupled indices `i` and
`i+1` on a fusion tree `f`, and returns the result as a dictionary of output trees and
corresponding coefficients.

The keyword `inv` determines whether index `i` will braid above or below index `i+1`, i.e.
applying `artin_braid(f′, i; inv = true)` to all the outputs `f′` of
`artin_braid(f, i; inv = false)` and collecting the results should yield a single fusion
tree with non-zero coefficient, namely `f` with coefficient `1`. This keyword has no effect
if `BraidingStyle(sectortype(f)) isa SymmetricBraiding`.
"""
function artin_braid(f::FusionTree{I, N}, i; inv::Bool = false) where {I, N}
    1 <= i < N ||
        throw(ArgumentError("Cannot swap outputs i=$i and i+1 out of only $N outputs"))
    uncoupled = f.uncoupled
    a, b = uncoupled[i], uncoupled[i + 1]
    uncoupled′ = TupleTools.setindex(uncoupled, b, i)
    uncoupled′ = TupleTools.setindex(uncoupled′, a, i + 1)
    coupled′ = f.coupled
    isdual′ = TupleTools.setindex(f.isdual, f.isdual[i], i + 1)
    isdual′ = TupleTools.setindex(isdual′, f.isdual[i + 1], i)
    inner = f.innerlines
    inner_extended = (uncoupled[1], inner..., coupled′)
    vertices = f.vertices
    oneT = one(sectorscalartype(I))

    if isunit(a) || isunit(b)
        # braiding with trivial sector: simple and always possible
        inner′ = inner
        vertices′ = vertices
        if i > 1 # we also need to alter innerlines and vertices
            inner′ = TupleTools.setindex(
                inner,
                inner_extended[isunit(a) ? (i + 1) : (i - 1)],
                i - 1
            )
            vertices′ = TupleTools.setindex(vertices′, vertices[i], i - 1)
            vertices′ = TupleTools.setindex(vertices′, vertices[i - 1], i)
        end
        f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner′, vertices′)
        return fusiontreedict(I)(f′ => oneT)
    end

    BraidingStyle(I) isa NoBraiding &&
        throw(SectorMismatch("Cannot braid sectors $(uncoupled[i]) and $(uncoupled[i + 1])"))

    if i == 1
        c = N > 2 ? inner[1] : coupled′
        if FusionStyle(I) isa MultiplicityFreeFusion
            R = oftype(oneT, (inv ? conj(Rsymbol(b, a, c)) : Rsymbol(a, b, c)))
            f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner, vertices)
            return fusiontreedict(I)(f′ => R)
        else # GenericFusion
            μ = vertices[1]
            Rmat = inv ? Rsymbol(b, a, c)' : Rsymbol(a, b, c)
            local newtrees
            for ν in axes(Rmat, 2)
                R = oftype(oneT, Rmat[μ, ν])
                iszero(R) && continue
                vertices′ = TupleTools.setindex(vertices, ν, 1)
                f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner, vertices′)
                if (@isdefined newtrees)
                    push!(newtrees, f′ => R)
                else
                    newtrees = fusiontreedict(I)(f′ => R)
                end
            end
            return newtrees
        end
    end
    # case i > 1: other naming convention
    b = uncoupled[i]
    d = uncoupled[i + 1]
    a = inner_extended[i - 1]
    c = inner_extended[i]
    e = inner_extended[i + 1]
    if FusionStyle(I) isa UniqueFusion
        c′ = first(a ⊗ d)
        coeff = oftype(
            oneT,
            if inv
                conj(Rsymbol(d, c, e) * Fsymbol(d, a, b, e, c′, c)) * Rsymbol(d, a, c′)
            else
                Rsymbol(c, d, e) * conj(Fsymbol(d, a, b, e, c′, c) * Rsymbol(a, d, c′))
            end
        )
        inner′ = TupleTools.setindex(inner, c′, i - 1)
        f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner′)
        return fusiontreedict(I)(f′ => coeff)
    elseif FusionStyle(I) isa SimpleFusion
        local newtrees
        for c′ in intersect(a ⊗ d, e ⊗ dual(b))
            coeff = oftype(
                oneT,
                if inv
                    conj(Rsymbol(d, c, e) * Fsymbol(d, a, b, e, c′, c)) * Rsymbol(d, a, c′)
                else
                    Rsymbol(c, d, e) * conj(Fsymbol(d, a, b, e, c′, c) * Rsymbol(a, d, c′))
                end
            )
            iszero(coeff) && continue
            inner′ = TupleTools.setindex(inner, c′, i - 1)
            f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner′)
            if (@isdefined newtrees)
                push!(newtrees, f′ => coeff)
            else
                newtrees = fusiontreedict(I)(f′ => coeff)
            end
        end
        return newtrees
    else # GenericFusion
        local newtrees
        for c′ in intersect(a ⊗ d, e ⊗ dual(b))
            Rmat1 = inv ? Rsymbol(d, c, e)' : Rsymbol(c, d, e)
            Rmat2 = inv ? Rsymbol(d, a, c′)' : Rsymbol(a, d, c′)
            Fmat = Fsymbol(d, a, b, e, c′, c)
            μ = vertices[i - 1]
            ν = vertices[i]
            for σ in 1:Nsymbol(a, d, c′)
                for λ in 1:Nsymbol(c′, b, e)
                    coeff = zero(oneT)
                    for ρ in 1:Nsymbol(d, c, e), κ in 1:Nsymbol(d, a, c′)
                        coeff += Rmat1[ν, ρ] * conj(Fmat[κ, λ, μ, ρ]) * conj(Rmat2[σ, κ])
                    end
                    iszero(coeff) && continue
                    vertices′ = TupleTools.setindex(vertices, σ, i - 1)
                    vertices′ = TupleTools.setindex(vertices′, λ, i)
                    inner′ = TupleTools.setindex(inner, c′, i - 1)
                    f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner′, vertices′)
                    if (@isdefined newtrees)
                        push!(newtrees, f′ => coeff)
                    else
                        newtrees = fusiontreedict(I)(f′ => coeff)
                    end
                end
            end
        end
        return newtrees
    end
end

# braid fusion tree
"""
    braid(f::FusionTree{<:Sector, N}, p::NTuple{N, Int}, levels::NTuple{N, Int})
    -> <:AbstractDict{typeof(t), <:Number}

Perform a braiding of the uncoupled indices of the fusion tree `f` and return the result as
a `<:AbstractDict` of output trees and corresponding coefficients. The braiding is
determined by specifying that the new sector at position `k` corresponds to the sector that
was originally at the position `i = p[k]`, and assigning to every index `i` of the original
fusion tree a distinct level or depth `levels[i]`. This permutation is then decomposed into
elementary swaps between neighbouring indices, where the swaps are applied as braids such
that if `i` and `j` cross, ``τ_{i,j}`` is applied if `levels[i] < levels[j]` and
``τ_{j,i}^{-1}`` if `levels[i] > levels[j]`. This does not allow to encode the most general
braid, but a general braid can be obtained by combining such operations.
"""
function braid(f::FusionTree{I, N}, p::NTuple{N, Int}, levels::NTuple{N, Int}) where {I, N}
    TupleTools.isperm(p) || throw(ArgumentError("not a valid permutation: $p"))
    if FusionStyle(I) isa UniqueFusion && BraidingStyle(I) isa SymmetricBraiding
        coeff = one(sectorscalartype(I))
        for i in 1:N
            for j in 1:(i - 1)
                if p[j] > p[i]
                    a, b = f.uncoupled[p[j]], f.uncoupled[p[i]]
                    coeff *= Rsymbol(a, b, first(a ⊗ b))
                end
            end
        end
        uncoupled′ = TupleTools._permute(f.uncoupled, p)
        coupled′ = f.coupled
        isdual′ = TupleTools._permute(f.isdual, p)
        f′ = FusionTree{I}(uncoupled′, coupled′, isdual′)
        return fusiontreedict(I)(f′ => coeff)
    else
        T = sectorscalartype(I)
        coeff = one(T)
        trees = FusionTreeDict(f => coeff)
        newtrees = empty(trees)
        for s in permutation2swaps(p)
            inv = levels[s] > levels[s + 1]
            for (f, c) in trees
                for (f′, c′) in artin_braid(f, s; inv)
                    newtrees[f′] = get(newtrees, f′, zero(coeff)) + c * c′
                end
            end
            l = levels[s]
            levels = TupleTools.setindex(levels, levels[s + 1], s)
            levels = TupleTools.setindex(levels, l, s + 1)
            trees, newtrees = newtrees, trees
            empty!(newtrees)
        end
        return trees
    end
end

# permute fusion tree
"""
    permute(f::FusionTree, p::NTuple{N, Int}) -> <:AbstractDict{typeof(t), <:Number}

Perform a permutation of the uncoupled indices of the fusion tree `f` and returns the result
as a `<:AbstractDict` of output trees and corresponding coefficients; this requires that
`BraidingStyle(sectortype(f)) isa SymmetricBraiding`.
"""
function permute(f::FusionTree{I, N}, p::NTuple{N, Int}) where {I, N}
    @assert BraidingStyle(I) isa SymmetricBraiding
    return braid(f, p, ntuple(identity, Val(N)))
end

# braid double fusion tree
"""
    braid((f₁, f₂)::FusionTreePair, (p1, p2)::Index2Tuple, (levels1, levels2)::Index2Tuple)
        -> <:AbstractDict{<:FusionTreePair{I, N₁, N₂}}, <:Number}

Input is a fusion-splitting tree pair that describes the fusion of a set of incoming
uncoupled sectors to a set of outgoing uncoupled sectors, represented using the splitting
tree `f₁` and fusion tree `f₂`, such that the incoming sectors `f₂.uncoupled` are fused to
`f₁.coupled == f₂.coupled` and then to the outgoing sectors `f₁.uncoupled`. Compute new
trees and corresponding coefficients obtained from repartitioning and braiding the tree such
that sectors `p1` become outgoing and sectors `p2` become incoming. The uncoupled indices in
splitting tree `f₁` and fusion tree `f₂` have levels (or depths) `levels1` and `levels2`
respectively, which determines how indices braid. In particular, if `i` and `j` cross,
``τ_{i,j}`` is applied if `levels[i] < levels[j]` and ``τ_{j,i}^{-1}`` if `levels[i] >
levels[j]`. This does not allow to encode the most general braid, but a general braid can
be obtained by combining such operations.
"""
function braid((f₁, f₂)::FusionTreePair, (p1, p2)::Index2Tuple,
               (levels1, levels2)::Index2Tuple)
    @assert length(f₁) + length(f₂) == length(p1) + length(p2)
    @assert length(f₁) == length(levels1) && length(f₂) == length(levels2)
    @assert TupleTools.isperm((p1..., p2...))
    return fsbraid(((f₁, f₂), (p1, p2), (levels1, levels2)))
end
const FSBraidKey{I, N₁, N₂} = Tuple{<:FusionTreePair{I}, Index2Tuple{N₁, N₂}, Index2Tuple}

@cached function fsbraid(key::FSBraidKey{I, N₁, N₂})::_fsdicttype(I, N₁, N₂) where {I, N₁, N₂}
    ((f₁, f₂), (p1, p2), (l1, l2)) = key
    p = linearizepermutation(p1, p2, length(f₁), length(f₂))
    levels = (l1..., reverse(l2)...)
    local newtrees
    for ((f, f0), coeff1) in repartition((f₁, f₂), N₁ + N₂)
        for (f′, coeff2) in braid(f, p, levels)
            for ((f₁′, f₂′), coeff3) in repartition((f′, f0), N₁)
                if @isdefined newtrees
                    newtrees[(f₁′, f₂′)] = get(newtrees, (f₁′, f₂′), zero(coeff3)) +
                        coeff1 * coeff2 * coeff3
                else
                    newtrees = fusiontreedict(I)((f₁′, f₂′) => coeff1 * coeff2 * coeff3)
                end
            end
        end
    end
    return newtrees
end

function CacheStyle(::typeof(fsbraid), k::FSBraidKey{I}) where {I <: Sector}
    if FusionStyle(I) isa UniqueFusion
        return NoCache()
    else
        return GlobalLRUCache()
    end
end

"""
    permute((f₁, f₂)::FusionTreePair, (p1, p2)::Index2Tuple)
        -> <:AbstractDict{<:FusionTreePair{I, N₁, N₂}}, <:Number}

Input is a double fusion tree that describes the fusion of a set of incoming uncoupled
sectors to a set of outgoing uncoupled sectors, represented using the individual trees of
outgoing (`t1`) and incoming sectors (`t2`) respectively (with identical coupled sector
`t1.coupled == t2.coupled`). Computes new trees and corresponding coefficients obtained from
repartitioning and permuting the tree such that sectors `p1` become outgoing and sectors
`p2` become incoming.
"""
function permute((f₁, f₂)::FusionTreePair, (p1, p2)::Index2Tuple)
    @assert BraidingStyle(sectortype(f₁)) isa SymmetricBraiding
    levels1 = ntuple(identity, length(f₁))
    levels2 = length(f₁) .+ ntuple(identity, length(f₂))
    return braid((f₁, f₂), (p1, p2), (levels1, levels2))
end
