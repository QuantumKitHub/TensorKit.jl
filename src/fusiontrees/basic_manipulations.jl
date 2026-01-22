# BASIC MANIPULATIONS:
#----------------------------------------------
# -> rewrite generic fusion tree in basis of fusion trees in standard form
# -> only depend on Fsymbol
"""
    split(f::FusionTree{I, N}, M::Int)
    -> (::FusionTree{I, M}, ::FusionTree{I, N-M+1})

Split a fusion tree into two. The first tree has as uncoupled sectors the first `M`
uncoupled sectors of the input tree `f`, whereas its coupled sector corresponds to the
internal sector between uncoupled sectors `M` and `M+1` of the original tree `f`. The
second tree has as first uncoupled sector that same internal sector of `f`, followed by
remaining `N-M` uncoupled sectors of `f`. It couples to the same sector as `f`. This
operation is the inverse of `join` in the sense that if `f == join(split(f, M)...)`
holds for all `M` between `0` and `N`, where `N` is the number of uncoupled sectors of `f`.

See also [`join`](@ref) and [`insertat`](@ref).
"""
@inline function split(f::FusionTree{I, N}, M::Int) where {I, N} # inline helps with constant propagation of M
    0 <= M <= N ||
        throw(ArgumentError("M should be between 0 and N = $N"))

    innerlines_extended = (f.uncoupled[1], f.innerlines..., f.coupled)
    vertices_extended = (1, f.vertices...)

    uncoupled1 = ntuple(n -> f.uncoupled[n], M)
    isdual1 = ntuple(n -> f.isdual[n], M)
    coupled1 = M == 0 ? leftunit(f.uncoupled[1]) : innerlines_extended[M]
    innerlines1 = ntuple(n -> f.innerlines[n], max(0, M - 2))
    vertices1 = ntuple(n -> f.vertices[n], max(0, M - 1))

    uncoupled2 = (coupled1, ntuple(n -> f.uncoupled[M + n], N - M)...)
    isdual2 = (false, ntuple(n -> f.isdual[M + n], N - M)...)
    coupled2 = f.coupled
    innerlines2 = ntuple(n -> innerlines_extended[M + n], max(0, N - M - 1))
    vertices2 = ntuple(n -> vertices_extended[M + n], N - M)

    f₁ = FusionTree{I}(uncoupled1, coupled1, isdual1, innerlines1, vertices1)
    f₂ = FusionTree{I}(uncoupled2, coupled2, isdual2, innerlines2, vertices2)
    return f₁, f₂
end

"""
    join(f₁::FusionTree{I, N₁}, f₂::FusionTree{I, N₂}) where {I, N₁, N₂}
    -> (::FusionTree{I, N₁ + N₂ - 1})

Join fusion trees `f₁` and `f₂` by connecting the coupled sector of `f₁` to the first
uncoupled sector of `f₂`. The resulting tree has uncoupled sectors given by those of `f₁`
followed the remaining uncoupled sectors (except for the first) of `f₂`. This
requires that `f₁.coupled == f₂.uncoupled[1]` and `f₂.isdual[1] == false`. This
operation is the inverse of split, in the sense that `f == join(split(f, M)...)`
holds for all `M` between `0` and `N`, where `N` is the number of uncoupled sectors of `f`.

See also [`split`](@ref) and [`insertat`](@ref).
"""
@inline function join(f₁::FusionTree{I, N₁}, f₂::FusionTree{I, N₂}) where {I, N₁, N₂}
    (f₁.coupled == f₂.uncoupled[1] && !f₂.isdual[1]) ||
        throw(SectorMismatch("cannot connect $(f₂.uncoupled[1]) to $(f₁.coupled)"))
    uncoupled = (f₁.uncoupled..., Base.tail(f₂.uncoupled)...)
    isdual = (f₁.isdual..., Base.tail(f₂.isdual)...)
    if N₁ == 0
        innerlines = N₂ <= 2 ? () : Base.tail(f₂.innerlines)
        vertices = N₂ <= 1 ? () : Base.tail(f₂.vertices)
    elseif N₁ == 1
        innerlines = f₂.innerlines
        vertices = f₂.vertices
    else
        innerlines = N₂ == 1 ? f₁.innerlines : (f₁.innerlines..., f₁.coupled, f₂.innerlines...)
        vertices = (f₁.vertices..., f₂.vertices...)
    end
    coupled = f₂.coupled
    return FusionTree{I}(uncoupled, coupled, isdual, innerlines, vertices)
end

"""
    VertexGetter(k::Int)

A helper struct to get both sectors left and right of the k-th uncoupled sectors,
as well as the corresponding vertex index.
"""
struct VertexGetter
    k::Int
    function VertexGetter(k::Int)
        k >= 2 || throw(ArgumentError("k must be at least 2"))
        return new(k)
    end
end
@inline function (vg::VertexGetter)(f::FusionTree)
    k = vg.k
    k <= length(f) ||
        throw(ArgumentError(lazy"k = $k exceeds number of uncoupled legs $(length(f))"))
    N = length(f)
    l = (k == 2) ? f.uncoupled[1] : f.innerlines[k - 2]
    r = (k == N) ? f.coupled : f.innerlines[k - 1]
    return l, r, f.vertices[k - 1]
end

"""
    function multi_associator(long::FusionTree{I,N}, short::FusionTree{I,N-1}) where {I, N}

Computes the associator coefficient for the following fusion tree transformation: 
```
        ╭ ⋯ ┴─╮
      ╭─┴─╮   |
    ╭─┴─╮ |   |
  ╭─┴─╮ | |   |
      | | |   |  =  coeff * ╭─┴─╮
      ╰┬╯ |   |
       ╰─┬╯   |
         ╰ ⋯ ┬╯
```
where the the upper splitting tree is given by `long` and the lower fusion tree by `short`.

When `FusionStyle(I) isa MultiplicityFreeFusion`, the coefficient is a scalar and the
splitting diagram on the right hand side is completely fixed as the only element in
the space `long.coupled → long.uncoupled[1] ⊗ short.coupled`.
In case of `FusionStyle(I) isa GenericFusion`, the coefficient is a vector, where the
different entries are associated with the different vertex indices on the splitting
tree on the right hand side.
"""
function multi_associator(long::FusionTree{I, N}, short) where {I, N}
    length(short) == N - 1 ||
        throw(DimensionMismatch("second fusion tree must have one less uncoupled leg"))
    uncoupled = long.uncoupled
    (uncoupled[2:end] == short.uncoupled && long.isdual[2:end] == short.isdual) ||
        return zero(sectorscalartype(typeof(long.coupled)))

    if FusionStyle(I) isa MultiplicityFreeFusion
        coeff = one(sectorscalartype(I))
    else
        coeff = fill(one(sectorscalartype(I)), 1)
    end
    a = uncoupled[1]
    for k in 2:(N - 1)
        c = uncoupled[k + 1]
        e, d, ν = VertexGetter(k + 1)(long)
        b, e′, κ = VertexGetter(k)(short)
        F = Fsymbol(a, b, c, d, e, e′)
        if FusionStyle(I) isa MultiplicityFreeFusion
            coeff *= F
        else
            if k == 2
                μ = long.vertices[1]
                coeff = transpose(view(F, μ:μ, ν, κ, :)) * coeff
            else
                coeff = transpose(view(F, :, ν, κ, :)) * coeff
            end
        end
    end
    return coeff
end

"""
    function multi_Fmove(tree::FusionTree{I,N}) where {I, N}

Computes the result of completely recoupling a fusion tree to split off
the first uncoupled sector

```
        ╭ ⋯ ┴─╮               ╭─ ⋯ ──┴─╮
      ╭─┴─╮   |               |      ╭─┴─╮
    ╭─┴─╮ |   |  =  ∑ coeff * |    ╭─┴─╮ |
  ╭─┴─╮ | |   |               |  ╭─┴─╮ | |
```

As the leftmost uncoupled sector `a = tree.uncoupled[1]` and the coupled sector `c = tree.copuled`
at the very top remain fixed, they are not returned. The result is returned as two arrays:
the first array contains the different fusion trees of the `N-1` uncoupled sectors on the right,
which is attached via some coupled sector `b` to the final fusion vertex. The second array contains
the corresponding expansion coefficients, either as scalars (if `FusionStyle(I) isa MultiplicityFreeFusion`)
or as vectors of length `Nsymbol(a, b, c)`, representing the different coefficients associated
with the different vertex labels `λ` of the topmost vertex.

See also [`multi_Fmove_inv`](@ref), [`multi_associator`](@ref).
"""
function multi_Fmove(f::FusionTree{I, N}) where {I, N}
    if FusionStyle(I) isa UniqueFusion
        coupled = N == 1 ? rightunit(f.uncoupled[1]) : (N == 2 ? f.uncoupled[2] : f.innerlines[end])
        f′ = FusionTree{I}(Base.tail(f.uncoupled), coupled, Base.tail(f.isdual))
        return (f′,), (multi_associator(f, f′),)
    end

    u = rightunit(f.coupled)
    T = typeof(Fsymbol(u, u, u, u, u, u)[1, 1, 1, 1])
    # sectorscalartype(I) may be different if there is also braiding
    # TODO: consider using _Fscalartype?

    if N == 1
        f′ = FusionTree{I}((), u, (), (), ())
        return [f′], FusionStyle(I) isa MultiplicityFreeFusion ? [one(T)] : [ones(T, 1)]
    elseif N == 2
        a, b = f.uncoupled
        c = f.coupled
        isdualb = f.isdual[2]
        f′ = FusionTree{I}((b,), b, (isdualb,), (), ())
        if FusionStyle(I) isa MultiplicityFreeFusion
            return [f′], [one(T)]
        else
            μ = f.vertices[1]
            coeff = zeros(T, Nsymbol(a, b, c))
            coeff[μ] = one(T)
            return [f′], [coeff]
        end
    else
        # Stage 1: generate all valid fusion trees
        a = f.uncoupled[1]
        f′ = FusionTree{I}(Base.tail(f.uncoupled), u, Base.tail(f.isdual), ntuple(n -> u, N - 3), ntuple(n -> 1, N - 2))
        # this is not a valid fusion tree; we generate trees along the way from left to right
        trees = [f′]
        treesprev = similar(trees, 0)
        for k in 2:(N - 1)
            treesprev, trees = trees, empty!(treesprev)
            treesprev = sort!(treesprev, by = VertexGetter(k))
            _, d, = VertexGetter(k + 1)(f)
            ād = dual(a) ⊗ d
            c = f.uncoupled[k + 1]
            b, = VertexGetter(k)(first(treesprev))
            b_current = b
            bc = b ⊗ c
            for tree in treesprev
                b, = VertexGetter(k)(tree)
                if b != b_current
                    bc = b ⊗ c
                    b_current = b
                end
                for e′ in intersect(bc, ād)
                    Nbce′ = Nsymbol(b, c, e′)
                    if k == N - 1
                        coupled = e′
                        innerlines = tree.innerlines
                    else
                        coupled = tree.coupled
                        innerlines = Base.setindex(tree.innerlines, e′, k - 1)
                    end
                    for μ in 1:Nbce′
                        vertices = Base.setindex(tree.vertices, μ, k - 1)
                        f′ = FusionTree{I}(tree.uncoupled, coupled, tree.isdual, innerlines, vertices)
                        push!(trees, f′)
                    end
                end
            end
        end
        # More expensive alternative: generate all fusion trees and filter
        # F = fusiontreetype(I, N - 1)
        # trees = Vector{F}(undef, 0)
        # for b̄ in dual(c) ⊗ a
        #     uncoupled′ = Base.tail(f.uncoupled)
        #     isdual′ = Base.tail(f.isdual)
        #     for f′ in fusiontrees(uncoupled′, dual(b̄), isdual′)
        #         keep = true
        #         for k in 2:(N - 1)
        #             e, d, = VertexGetter(k + 1)(f)
        #             b, e′, = VertexGetter(k)(f′)
        #             if !(e ∈ a ⊗ b && d ∈ a ⊗ e′)
        #                 keep = false
        #                 break
        #             end
        #         end
        #         keep && push!(trees, f′)
        #     end
        # end

        # Stage 2: compute corresponding expansion coefficients from left to right
        coeffs = FusionStyle(I) isa MultiplicityFreeFusion ?
            Vector{T}(undef, length(trees)) : Vector{Vector{T}}(undef, length(trees))
        a = f.uncoupled[1]
        b = f.uncoupled[2]
        c = f.uncoupled[3]
        _, e, μ = VertexGetter(2)(f)
        _, d, ν = VertexGetter(3)(f)
        p = sortperm(trees, by = VertexGetter(2)) # first return value of VertexGetter(2) is 'a' which is constant
        tree = trees[p[1]]
        _, e′, = VertexGetter(2)(tree)
        e′_current = e′
        F_current = Fsymbol(a, b, c, d, e, e′)
        for i in p
            _, e′, κ = VertexGetter(2)(trees[i])
            if e′ != e′_current
                F_current = Fsymbol(a, b, c, d, e, e′)
                e′_current = e′
            end
            F = F_current
            if FusionStyle(I) isa MultiplicityFreeFusion
                coeffs[i] = F
            else
                coeffs[i] = F[μ, ν, κ, :]
            end
        end
        for k in 3:(N - 1)
            c = f.uncoupled[k + 1]
            e = d
            _, d, ν = VertexGetter(k + 1)(f)
            p = sortperm!(p, trees, by = VertexGetter(k))
            tree = trees[p[1]]
            b, e′, = VertexGetter(k)(tree)
            b_current = b
            e′_current = e′
            F_current = Fsymbol(a, b, c, d, e, e′)
            for i in p
                b, e′, κ = VertexGetter(k)(trees[i])
                if b != b_current || e′ != e′_current
                    F_current = Fsymbol(a, b, c, d, e, e′)
                    b_current = b
                    e′_current = e′
                end
                F = F_current
                if FusionStyle(I) isa MultiplicityFreeFusion
                    coeffs[i] *= F
                else
                    coeffs[i] = transpose(view(F, :, ν, κ, :)) * coeffs[i]
                end
            end
        end
        # TODO: it would be more uniform to return this as a dictionary
        # Would that extra step create significant extra overhead?
        return trees, coeffs
    end
end

"""
    function multi_Fmove_inv(a, c, tree::FusionTree{I, N}) where {I, N}

Computes the expansion of fusing a left uncoupled sector `a` with an existing fusion tree
`tree` with coupled sector `b = tree.coupled` to a coupled sector `c`, and recoupling the
result into a linear combination of trees in standard form.
```
  ╭─ ⋯ ──┴─╮                       ╭ ⋯ ┴─╮
  |      ╭─┴─╮                   ╭─┴─╮   |
  |    ╭─┴─╮ |  =  ∑ coeff *   ╭─┴─╮ |   |
  |  ╭─┴─╮ | |               ╭─┴─╮ | |   |
```

The result is returned as two arrays: the first array contains the different fusion trees of
the `N+1` uncoupled sectors. The second array contains the corresponding expansion coefficients,
either as scalars (if `FusionStyle(I) isa MultiplicityFreeFusion`) or as vectors of length
`Nsymbol(a, b, c)`, representing the different coefficients associated with the different
possible vertex labels `λ` of the topmost vertex in the left hand side.
"""
function multi_Fmove_inv(a, c, f::FusionTree{I, N}) where {I, N}
    b = f.coupled
    c ∈ a ⊗ b ||
        throw(SectorMismatch("cannot fuse sectors $a and $b to $c"))

    if FusionStyle(I) isa UniqueFusion
        f′ = FusionTree{I}((a, f.uncoupled...), c, (false, f.isdual...))
        return (f′,), (conj(multi_associator(f′, f)),)
    end

    u = rightunit(c)
    T = typeof(Fsymbol(u, u, u, u, u, u)[1, 1, 1, 1])
    F = fusiontreetype(I, N + 1)
    # sectorscalartype(I) may be different if there is also braiding
    # TODO: consider using _Fscalartype?
    if N == 0
        f′ = FusionTree{I}((a,), c, (false,), (), ())
        return F[f′], FusionStyle(I) isa MultiplicityFreeFusion ? [one(T)] : [ones(T, 1)]
    elseif N == 1
        Nabc = Nsymbol(a, b, c)
        trees = Vector{F}(undef, Nabc)
        coeffs = FusionStyle(I) isa MultiplicityFreeFusion ? Vector{T}(undef, Nabc) : Vector{Vector{T}}(undef, Nabc)
        if FusionStyle(I) isa MultiplicityFreeFusion
            trees[1] = FusionTree{I}((a, f.uncoupled[1]), c, (false, f.isdual[1]), ())
            coeffs[1] = one(T)
        else
            for μ in 1:Nabc
                trees[μ] = FusionTree{I}((a, f.uncoupled[1]), c, (false, f.isdual[1]), (μ,))
                coeff = zeros(T, Nsymbol(a, b, c))
                coeff[μ] = one(T)
                coeffs[μ] = coeff
            end
        end
        return trees, coeffs
    else
        # Stage 1: generate all valid fusion trees
        f′ = FusionTree{I}((a, f.uncoupled...), c, (false, f.isdual...), ntuple(n -> u, N - 1), ntuple(n -> 1, N))
        # this is not a valid fusion tree; we generate trees along the way from right to left
        trees = [f′]
        treesprev = similar(trees, 0)
        for k in N:-1:2
            c = f.uncoupled[k]
            b, e′, = VertexGetter(k)(f)
            ab = a ⊗ b
            treesprev, trees = trees, empty!(treesprev)
            treesprev = sort!(treesprev, by = VertexGetter(k + 1))
            _, d, = VertexGetter(k + 1)(first(treesprev))
            d_current = d
            dc̄ = d ⊗ dual(c)
            for tree in treesprev
                _, d, = VertexGetter(k + 1)(tree)
                if d != d_current
                    dc̄ = d ⊗ dual(c)
                    d_current = d
                end
                for e in intersect(ab, dc̄)
                    Necd = Nsymbol(e, c, d)
                    Nabc = k == 2 ? Nsymbol(a, b, e) : 1 # only set μ on final step
                    innerlines = Base.setindex(tree.innerlines, e, k - 1)
                    for ν in 1:Necd, μ in 1:Nabc
                        vertices = Base.setindex(tree.vertices, ν, k)
                        vertices = Base.setindex(vertices, μ, k - 1)
                        f′ = FusionTree{I}(tree.uncoupled, tree.coupled, tree.isdual, innerlines, vertices)
                        push!(trees, f′)
                    end
                end
            end
        end
        # More expensive alternative: generate all fusion trees and filter
        # for f′ in fusiontrees((a, f.uncoupled...), c, (false, f.isdual...))
        #     keep = true
        #     for k in 2:N
        #         b, e′, = VertexGetter(k)(f)
        #         e, d, = VertexGetter(k + 1)(f′)
        #         if !(e ∈ a ⊗ b && d ∈ a ⊗ e′)
        #             keep = false
        #             break
        #         end
        #     end
        #     keep && push!(trees, f′)
        # end

        # Stage 2: compute corresponding expansion coefficients from left to right
        coeffs = FusionStyle(I) isa MultiplicityFreeFusion ?
            Vector{T}(undef, length(trees)) : Vector{Vector{T}}(undef, length(trees))
        b = f.uncoupled[1]
        c = f.uncoupled[2]
        _, e′, κ = VertexGetter(2)(f)
        p = sortperm(trees, by = VertexGetter(3))
        tree = trees[p[1]]
        e, d, = VertexGetter(3)(tree)
        e_current = e
        d_current = d
        F_current = Fsymbol(a, b, c, d, e, e′)
        for i in p
            μ = trees[i].vertices[1]
            e, d, ν = VertexGetter(3)(trees[i])
            if e != e_current || d != d_current
                F_current = Fsymbol(a, b, c, d, e, e′)
                e_current = e
                d_current = d
            end
            F = F_current
            if FusionStyle(I) isa MultiplicityFreeFusion
                coeffs[i] = conj(F)
            else
                coeffs[i] = conj!(F[μ, ν, κ, :])
            end
        end
        for k in 3:N
            c = f.uncoupled[k]
            b, e′, κ = VertexGetter(k)(f)
            p = sortperm!(p, trees, by = VertexGetter(k + 1))
            tree = trees[p[1]]
            e, d, = VertexGetter(k + 1)(tree)
            e_current = e
            d_current = d
            F_current = Fsymbol(a, b, c, d, e, e′)
            for i in p
                e, d, ν = VertexGetter(k + 1)(trees[i])
                if e != e_current || d != d_current
                    F_current = Fsymbol(a, b, c, d, e, e′)
                    e_current = e
                    d_current = d
                end
                F = F_current
                if FusionStyle(I) isa MultiplicityFreeFusion
                    coeffs[i] *= F
                else
                    coeffs[i] = view(F, :, ν, κ, :)' * coeffs[i]
                end
            end
        end
        # TODO: it would be more uniform to return this as a dictionary
        # Would that extra step create significant extra overhead?
        return trees, coeffs
    end
end

"""
    insertat(f::FusionTree{I, N₁}, i::Int, f₂::FusionTree{I, N₂})
    -> <:AbstractDict{<:FusionTree{I, N₁+N₂-1}, <:Number}

Attach a fusion tree `f₂` to the uncoupled leg `i` of the fusion tree `f₁` and bring it
into a linear combination of fusion trees in standard form. This requires that
`f₂.coupled == f₁.uncoupled[i]` and `f₁.isdual[i] == false`.
"""
@inline function insertat(f₁::FusionTree{I, N₁}, i, f₂::FusionTree{I, N₂}) where {I, N₁, N₂}
    (f₁.uncoupled[i] == f₂.coupled && !f₁.isdual[i]) ||
        throw(SectorMismatch("cannot connect $(f₂.uncoupled) to $(f₁.uncoupled[i])"))

    F = fusiontreetype(I, N₁ + N₂ - 1)
    u = rightunit(f₁.coupled)
    T = typeof(Fsymbol(u, u, u, u, u, u)[1, 1, 1, 1])
    # sectorscalartype(I) may be different if there is also braiding
    # TODO: consider using _Fscalartype?

    i == 1 && return fusiontreedict(I){F, T}(join(f₂, f₁) => one(T))

    innerlines_extended = (f₁.uncoupled[1], f₁.innerlines..., f₁.coupled)
    as = (innerlines_extended[i - 1], f₂.uncoupled...)
    c = innerlines_extended[i]
    fleft, = split(f₁, i - 1)
    _, fright = split(f₁, i)
    a, c, λ = VertexGetter(i)(f₁)
    middletrees, middlecoeffs = multi_Fmove_inv(a, c, f₂)
    if FusionStyle(I) isa UniqueFusion
        fmiddle = only(middletrees)
        coeff = only(middlecoeffs)
        f′ = join(join(fleft, fmiddle), fright)
        return fusiontreedict(I){F, T}(f′ => coeff)
    else
        newtrees = fusiontreedict(I){F, T}()
        for (fmiddle, coeff_middle) in zip(middletrees, middlecoeffs)
            coeff = coeff_middle[λ]
            iszero(coeff) && continue
            f′ = join(join(fleft, fmiddle), fright)
            push!(newtrees, f′ => coeff)
        end
        return newtrees
    end
end


"""
    merge(f₁::FusionTree{I, N₁}, f₂::FusionTree{I, N₂}, c::I, μ = 1)
    -> <:AbstractDict{<:FusionTree{I, N₁+N₂}, <:Number}

Merge two fusion trees together to a linear combination of fusion trees whose uncoupled
sectors are those of `f₁` followed by those of `f₂`, and where the two coupled sectors of
`f₁` and `f₂` are further fused to `c`. In case of `FusionStyle(I) == GenericFusion()`,
also a degeneracy label `μ` for the fusion of the coupled sectors of `f₁` and `f₂` to
`c` needs to be specified.
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
    f = join(f₁, f₀)
    return insertat(f, N₁ + 1, f₂)
end
function merge(f₁::FusionTree{I, 0}, f₂::FusionTree{I, 0}, c::I, μ) where {I}
    (f₁.coupled == c && μ == 1 && f₂.coupled == rightunit(c)) ||
        throw(SectorMismatch("cannot fuse sectors $(f₁.coupled) and $(f₂.coupled) to $c"))

    u = rightunit(f₁.coupled)
    T = typeof(Fsymbol(u, u, u, u, u, u)[1, 1, 1, 1])
    return fusiontreedict(I)(f₁ => one(T))
end

# flip a duality flag of a fusion tree
# TODO: move to duality or braiding manipulations (requires duality and twist)
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

# Legacy implementation of insertat: to be removed if confirmed slower
# function insertat_old(f₁::FusionTree{I}, i::Int, f₂::FusionTree{I, 0}) where {I}
#     # this actually removes uncoupled line i, which should be trivial
#     (f₁.uncoupled[i] == f₂.coupled && !f₁.isdual[i]) ||
#         throw(SectorMismatch("cannot connect $(f₂.uncoupled) to $(f₁.uncoupled[i])"))
#     u = rightunit(f₂.coupled)
#     T = typeof(Fsymbol(u, u, u, u, u, u)[1, 1, 1, 1])
#     coeff = one(T)
#     # sectorscalartype(I) may be different if there is also braiding
#     # TODO: consider using _Fscalartype?

#     uncoupled = TupleTools.deleteat(f₁.uncoupled, i)
#     coupled = f₁.coupled
#     isdual = TupleTools.deleteat(f₁.isdual, i)
#     if length(uncoupled) <= 2
#         inner = ()
#     else
#         inner = TupleTools.deleteat(f₁.innerlines, max(1, i - 2))
#     end
#     if length(uncoupled) <= 1
#         vertices = ()
#     else
#         vertices = TupleTools.deleteat(f₁.vertices, max(1, i - 1))
#     end
#     f = FusionTree(uncoupled, coupled, isdual, inner, vertices)
#     return fusiontreedict(I)(f => coeff)
# end
# function insertat_old(f₁::FusionTree{I}, i, f₂::FusionTree{I, 1}) where {I}
#     # identity operation
#     (f₁.uncoupled[i] == f₂.coupled && !f₁.isdual[i]) ||
#         throw(SectorMismatch("cannot connect $(f₂.uncoupled) to $(f₁.uncoupled[i])"))
#     u = rightunit(f₂.coupled)
#     T = typeof(Fsymbol(u, u, u, u, u, u)[1, 1, 1, 1])
#     coeff = one(T)
#     # sectorscalartype(I) may be different if there is also braiding
#     # TODO: consider using _Fscalartype?
#     isdual′ = TupleTools.setindex(f₁.isdual, f₂.isdual[1], i)
#     f = FusionTree{I}(f₁.uncoupled, f₁.coupled, isdual′, f₁.innerlines, f₁.vertices)
#     return fusiontreedict(I)(f => coeff)
# end
# function insertat_old(f₁::FusionTree{I}, i, f₂::FusionTree{I, 2}) where {I}
#     # elementary building block,
#     (f₁.uncoupled[i] == f₂.coupled && !f₁.isdual[i]) ||
#         throw(SectorMismatch("cannot connect $(f₂.uncoupled) to $(f₁.uncoupled[i])"))
#     uncoupled = f₁.uncoupled
#     coupled = f₁.coupled
#     inner = f₁.innerlines
#     b, c = f₂.uncoupled
#     isdual = f₁.isdual
#     isdualb, isdualc = f₂.isdual
#     if i == 1
#         uncoupled′ = (b, c, tail(uncoupled)...)
#         isdual′ = (isdualb, isdualc, tail(isdual)...)
#         inner′ = (uncoupled[1], inner...)
#         vertices′ = (f₂.vertices..., f₁.vertices...)
#         coeff = one(sectorscalartype(I))
#         f′ = FusionTree(uncoupled′, coupled, isdual′, inner′, vertices′)
#         return fusiontreedict(I)(f′ => coeff)
#     end
#     uncoupled′ = TupleTools.insertafter(TupleTools.setindex(uncoupled, b, i), i, (c,))
#     isdual′ = TupleTools.insertafter(TupleTools.setindex(isdual, isdualb, i), i, (isdualc,))
#     inner_extended = (uncoupled[1], inner..., coupled)
#     a = inner_extended[i - 1]
#     d = inner_extended[i]
#     e′ = uncoupled[i]
#     if FusionStyle(I) isa MultiplicityFreeFusion
#         local newtrees
#         for e in a ⊗ b
#             coeff = conj(Fsymbol(a, b, c, d, e, e′))
#             iszero(coeff) && continue
#             inner′ = TupleTools.insertafter(inner, i - 2, (e,))
#             f′ = FusionTree(uncoupled′, coupled, isdual′, inner′)
#             if @isdefined newtrees
#                 push!(newtrees, f′ => coeff)
#             else
#                 newtrees = fusiontreedict(I)(f′ => coeff)
#             end
#         end
#         return newtrees
#     else
#         local newtrees
#         κ = f₂.vertices[1]
#         λ = f₁.vertices[i - 1]
#         for e in a ⊗ b
#             inner′ = TupleTools.insertafter(inner, i - 2, (e,))
#             Fmat = Fsymbol(a, b, c, d, e, e′)
#             for μ in axes(Fmat, 1), ν in axes(Fmat, 2)
#                 coeff = conj(Fmat[μ, ν, κ, λ])
#                 iszero(coeff) && continue
#                 vertices′ = TupleTools.setindex(f₁.vertices, ν, i - 1)
#                 vertices′ = TupleTools.insertafter(vertices′, i - 2, (μ,))
#                 f′ = FusionTree(uncoupled′, coupled, isdual′, inner′, vertices′)
#                 if @isdefined newtrees
#                     push!(newtrees, f′ => coeff)
#                 else
#                     newtrees = fusiontreedict(I)(f′ => coeff)
#                 end
#             end
#         end
#         return newtrees
#     end
# end
# function insertat_old(f₁::FusionTree{I, N₁}, i, f₂::FusionTree{I, N₂}) where {I, N₁, N₂}
#     F = fusiontreetype(I, N₁ + N₂ - 1)
#     (f₁.uncoupled[i] == f₂.coupled && !f₁.isdual[i]) ||
#         throw(SectorMismatch("cannot connect $(f₂.uncoupled) to $(f₁.uncoupled[i])"))
#     T = sectorscalartype(I)
#     coeff = one(T)
#     if length(f₁) == 1
#         return fusiontreedict(I){F, T}(f₂ => coeff)
#     end
#     if i == 1
#         uncoupled = (f₂.uncoupled..., tail(f₁.uncoupled)...)
#         isdual = (f₂.isdual..., tail(f₁.isdual)...)
#         inner = (f₂.innerlines..., f₂.coupled, f₁.innerlines...)
#         vertices = (f₂.vertices..., f₁.vertices...)
#         coupled = f₁.coupled
#         f′ = FusionTree(uncoupled, coupled, isdual, inner, vertices)
#         return fusiontreedict(I){F, T}(f′ => coeff)
#     else # recursive definition
#         N2 = length(f₂)
#         f₂′, f₂′′ = split(f₂, N2 - 1)
#         local newtrees::fusiontreedict(I){F, T}
#         for (f, coeff) in insertat_old(f₁, i, f₂′′)
#             for (f′, coeff′) in insertat_old(f, i, f₂′)
#                 if @isdefined newtrees
#                     coeff′′ = coeff * coeff′
#                     newtrees[f′] = get(newtrees, f′, zero(coeff′′)) + coeff′′
#                 else
#                     newtrees = fusiontreedict(I){F, T}(f′ => coeff * coeff′)
#                 end
#             end
#         end
#         return newtrees
#     end
# end
