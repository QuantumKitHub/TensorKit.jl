# ELEMENTARY DUALITY MANIPULATIONS: A- and B-moves
#---------------------------------------------------------
# -> elementary manipulations that depend on the duality (rigidity) and pivotal structure
# -> planar manipulations that do not require braiding, everything is in Fsymbol (A/Bsymbol)
# -> B-move (bendleft, bendright) is simple in standard basis
# -> A-move (foldleft, foldright) is complicated, needs to be reexpressed in standard form

@doc """
    bendright((f₁, f₂)::FusionTreePair) -> (f₃, f₄) => coeff
    bendright(src::FusionTreeBlock) -> dst => coeffs

Map the final splitting vertex `a ⊗ b ← c` of `src` to a fusion vertex `a ← c ⊗ dual(b)` in `dst`.
For `FusionStyle(src) === UniqueFusion()`, both `src` and `dst` are simple `FusionTreePair`s, and the
transformation consists of a single coefficient `coeff`.
For generic `FusionStyle`s, the input and output consist of `FusionTreeBlock`s that bundle together
all trees with the same uncoupled charges, and `coeffs` now forms a transformation matrix.

```
    ╰─┬─╯ |  | |   ╰─┬─╯ |  |  |
      ╰─┬─╯  | |     ╰─┬─╯  |  |
        ╰ ⋯ ┬╯ |       ╰ ⋯ ┬╯  |
            |  | →         ╰─┬─╯
        ╭ ⋯ ┴╮ |         ╭ ⋯ ╯
      ╭─┴─╮  | |       ╭─┴─╮
    ╭─┴─╮ |  ╰─╯     ╭─┴─╮ |
```

See also [`bendleft`](@ref).
""" bendright

function bendright((f₁, f₂)::FusionTreePair)
    I = sectortype((f₁, f₂))
    @assert FusionStyle(I) === UniqueFusion()
    N₁, N₂ = numout((f₁, f₂)), numin((f₁, f₂))
    @assert N₁ > 0
    c = f₁.coupled
    a = N₁ == 1 ? leftunit(f₁.uncoupled[1]) : (N₁ == 2 ? f₁.uncoupled[1] : f₁.innerlines[end])
    b = f₁.uncoupled[N₁]

    # construct the new fusiontree pair
    uncoupled1 = TupleTools.front(f₁.uncoupled)
    isdual1 = TupleTools.front(f₁.isdual)
    inner1 = N₁ > 2 ? TupleTools.front(f₁.innerlines) : ()
    vertices1 = N₁ > 1 ? TupleTools.front(f₁.vertices) : ()
    f₁′ = FusionTree{I}(uncoupled1, a, isdual1, inner1, vertices1)

    uncoupled2 = (f₂.uncoupled..., dual(b))
    isdual2 = (f₂.isdual..., !(f₁.isdual[N₁]))
    inner2 = N₂ > 1 ? (f₂.innerlines..., c) : ()
    vertices2 = N₂ > 0 ? (f₂.vertices..., 1) : ()
    f₂′ = FusionTree{I}(uncoupled2, a, isdual2, inner2, vertices2)

    # compute the coefficient
    coeff₀ = sqrtdim(c) * invsqrtdim(a)
    f₁.isdual[N₁] && (coeff₀ *= conj(frobenius_schur_phase(dual(b))))
    coeff = coeff₀ * Bsymbol(a, b, c)

    return (f₁′, f₂′) => coeff
end
function bendright(src::FusionTreeBlock)
    uncoupled_dst = (
        TupleTools.front(src.uncoupled[1]),
        (src.uncoupled[2]..., dual(src.uncoupled[1][end])),
    )
    isdual_dst = (
        TupleTools.front(src.isdual[1]),
        (src.isdual[2]..., !(src.isdual[1][end])),
    )
    I = sectortype(src)
    N₁ = numout(src)
    N₂ = numin(src)
    @assert N₁ > 0

    dst = FusionTreeBlock{I}(uncoupled_dst, isdual_dst; sizehint = length(src))
    indexmap = treeindex_map(dst)
    U = zeros(sectorscalartype(I), length(dst), length(src))

    for (col, (f₁, f₂)) in enumerate(fusiontrees(src))
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
            row = indexmap[treeindex_data((f₁′, f₂′))]
            @inbounds U[row, col] = coeff
        else
            Bmat = Bsymbol(a, b, c)
            μ = N₁ > 1 ? f₁.vertices[end] : 1
            for ν in axes(Bmat, 2)
                coeff = coeff₀ * Bmat[μ, ν]
                iszero(coeff) && continue
                vertices2 = N₂ > 0 ? (f₂.vertices..., ν) : ()
                f₂′ = FusionTree(uncoupled2, a, isdual2, inner2, vertices2)
                row = indexmap[treeindex_data((f₁′, f₂′))]
                @inbounds U[row, col] = coeff
            end
        end
    end

    return dst => U
end

@doc """
    bendleft((f₁, f₂)::FusionTreePair) -> (f₃, f₄) => coeff
    bendleft(src::FusionTreeBlock) -> dst => coeffs

Map the final fusion vertex `a ← c ⊗ dual(b)` of `src` to a splitting vertex `a ⊗ b ← c` in `dst`.
For `FusionStyle(src) === UniqueFusion()`, both `src` and `dst` are simple `FusionTreePair`s, and the
transformation consists of a single coefficient `coeff`.
For generic `FusionStyle`s, the input and output consist of `FusionTreeBlock`s that bundle together
all trees with the same uncoupled charges, and `coeffs` now forms a transformation matrix.

```
    ╰─┬─╯ |  ╭─╮     ╰─┬─╯ |
      ╰─┬─╯  | |       ╰─┬─╯ 
        ╰ ⋯ ┬╯ |         ╰ ⋯ ╮
            |  | →         ╭─┴─╮
        ╭ ⋯ ┴╮ |       ╭ ⋯ ┴╮  |
      ╭─┴─╮  | |     ╭─┴─╮  |  |
    ╭─┴─╮ |  | |   ╭─┴─╮ |  |  |
```

See also [`bendright`](@ref).
""" bendleft

function bendleft((f₁, f₂)::FusionTreePair)
    @assert FusionStyle((f₁, f₂)) === UniqueFusion()
    (f₂′, f₁′), coeff = bendright((f₂, f₁))
    return (f₁′, f₂′) => conj(coeff)
end

# !! note that this is more or less a copy of bendright through
# (f1, f2) => conj(coeff) for ((f2, f1), coeff) in bendleft(src)
function bendleft(src::FusionTreeBlock)
    uncoupled_dst = (
        (src.uncoupled[1]..., dual(src.uncoupled[2][end])),
        TupleTools.front(src.uncoupled[2]),
    )
    isdual_dst = (
        (src.isdual[1]..., !(src.isdual[2][end])),
        TupleTools.front(src.isdual[2]),
    )
    I = sectortype(src)
    N₁ = numin(src)
    N₂ = numout(src)
    @assert N₁ > 0

    dst = FusionTreeBlock{I}(uncoupled_dst, isdual_dst; sizehint = length(src))
    indexmap = treeindex_map(dst)
    U = zeros(sectorscalartype(I), length(dst), length(src))

    for (col, (f₂, f₁)) in enumerate(fusiontrees(src))
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
            row = indexmap[treeindex_data((f₂′, f₁′))]
            @inbounds U[row, col] = conj(coeff)
        else
            Bmat = Bsymbol(a, b, c)
            μ = N₁ > 1 ? f₁.vertices[end] : 1
            for ν in axes(Bmat, 2)
                coeff = coeff₀ * Bmat[μ, ν]
                iszero(coeff) && continue
                vertices2 = N₂ > 0 ? (f₂.vertices..., ν) : ()
                f₂′ = FusionTree(uncoupled2, a, isdual2, inner2, vertices2)
                row = indexmap[treeindex_data((f₂′, f₁′))]
                @inbounds U[row, col] = conj(coeff)
            end
        end
    end

    return dst => U
end


@doc """
    foldright((f₁, f₂)::FusionTreePair) -> (f₃, f₄) => coeff
    foldright(src::FusionTreeBlock) -> dst => coeffs

Map the first splitting vertex `a ⊗ b ← c` of `src` to a fusion vertex `a ← c ⊗ dual(b)` in `dst`.
For `FusionStyle(src) === UniqueFusion()`, both `src` and `dst` are simple `FusionTreePair`s, and the
transformation consists of a single coefficient `coeff`.
For generic `FusionStyle`s, the input and output consist of `FusionTreeBlock`s that bundle together
all trees with the same uncoupled charges, and `coeffs` now forms a transformation matrix.

```
    | ╰─┬─╯ |  |   ╰─┬─╯ | |  |
    |   ╰─┬─╯  |     ╰─┬─╯ |  |
    |     ╰ ⋯ ┬╯       ╰─┬─╯  |
    |         |  →       ╰ ⋯ ┬╯
    |     ╭ ⋯ ┴╮             |
    |   ╭─┴─╮  |        ╭─ ⋯ ┴╮
    ╰───┴─╮ |  |      ╭─┴─╮   |
```

See also [`foldleft`](@ref).
""" foldright

function foldright((f₁, f₂)::FusionTreePair)
    I = sectortype((f₁, f₂))
    @assert FusionStyle(I) === UniqueFusion()
    @assert length(f₁) > 0

    # compute new trees
    a = f₁.uncoupled[1]
    isduala = f₁.isdual[1]
    c1 = dual(a)
    c2 = f₁.coupled
    c = first(c1 ⊗ c2)
    fl = FusionTree{I}(Base.tail(f₁.uncoupled), c, Base.tail(f₁.isdual))
    fr = FusionTree{I}((c1, f₂.uncoupled...), c, (!isduala, f₂.isdual...))

    # compute new coefficients
    factor = sqrtdim(a)
    isduala || (factor *= conj(frobenius_schur_phase(a)))

    return (fl, fr) => factor
end

function foldright(src::FusionTreeBlock)
    uncoupled_dst = (
        Base.tail(src.uncoupled[1]),
        (dual(first(src.uncoupled[1])), src.uncoupled[2]...),
    )
    isdual_dst = (Base.tail(src.isdual[1]), (!first(src.isdual[1]), src.isdual[2]...))
    I = sectortype(src)
    N₁ = numout(src)
    @assert N₁ > 0

    dst = FusionTreeBlock{I}(uncoupled_dst, isdual_dst; sizehint = length(src))
    indexmap = treeindex_map(dst)
    U = zeros(sectorscalartype(I), length(dst), length(src))

    for (col, (f₁, f₂)) in enumerate(fusiontrees(src))
        # map first splitting vertex (a, b)<-c to fusion vertex b<-(dual(a), c)
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
            row = indexmap[treeindex_data((fl, fr))]
            @inbounds U[row, col] = factor
        else
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
                    frs_coeffs = insertat(fc, 2, f₂)
                    for (fl′, coeff1) in insertat(fc, 2, f₁)
                        N₁ > 1 && !isone(fl′.innerlines[1]) && continue
                        coupled = fl′.coupled
                        uncoupled = Base.tail(Base.tail(fl′.uncoupled))
                        isdual = Base.tail(Base.tail(fl′.isdual))
                        inner = N₁ <= 3 ? () : Base.tail(Base.tail(fl′.innerlines))
                        vertices = N₁ <= 2 ? () : Base.tail(Base.tail(fl′.vertices))
                        fl = FusionTree{I}(uncoupled, coupled, isdual, inner, vertices)
                        for (fr, coeff2) in frs_coeffs
                            coeff = factor * coeff1 * conj(coeff2)
                            row = indexmap[treeindex_data((fl, fr))]
                            @inbounds U[row, col] = coeff
                        end
                    end
                end
            end
        end
    end

    return dst => U
end

@doc """
    foldleft((f₁, f₂)::FusionTreePair) -> (f₃, f₄) => coeff
    foldleft(src::FusionTreeBlock) -> dst => coeffs

Map the first fusion vertex `a ← c ⊗ dual(b)` of `src` to a splitting vertex `a ⊗ b ← c` in `dst`.
For `FusionStyle(src) === UniqueFusion()`, both `src` and `dst` are simple `FusionTreePair`s, and the
transformation consists of a single coefficient `coeff`.
For generic `FusionStyle`s, the input and output consist of `FusionTreeBlock`s that bundle together
all trees with the same uncoupled charges, and `coeffs` now forms a transformation matrix.

```
    ╭───┬─╯ |  |       ╰─┬─╯  |
    |   ╰─┬─╯  |         ╰ ⋯ ┬╯ 
    |     ╰ ⋯ ┬╯             |
    |         |  →       ╭ ⋯ ┴╮
    |     ╭ ⋯ ┴╮       ╭─┴─╮  |
    |   ╭─┴─╮  |     ╭─┴─╮ |  |
    | ╭─┴─╮ |  |   ╭─┴─╮ | |  |
```

See also [`foldright`](@ref).
""" foldleft

function foldleft((f₁, f₂)::FusionTreePair)
    @assert FusionStyle((f₁, f₂)) === UniqueFusion()
    (f₂′, f₁′), coeff = foldright((f₂, f₁))
    return (f₁′, f₂′) => conj(coeff)
end

# !! note that this is more or less a copy of foldright through
# (f1, f2) => conj(coeff) for ((f2, f1), coeff) in foldright(src)
function foldleft(src::FusionTreeBlock)
    uncoupled_dst = (
        (dual(first(src.uncoupled[2])), src.uncoupled[1]...),
        Base.tail(src.uncoupled[2]),
    )
    isdual_dst = (
        (!first(src.isdual[2]), src.isdual[1]...),
        Base.tail(src.isdual[2]),
    )
    I = sectortype(src)
    N₁ = numin(src)
    N₂ = numout(src)
    @assert N₁ > 0

    dst = FusionTreeBlock{I}(uncoupled_dst, isdual_dst; sizehint = length(src))
    indexmap = treeindex_map(dst)
    U = zeros(sectorscalartype(I), length(dst), length(src))

    for (col, (f₂, f₁)) in enumerate(fusiontrees(src))
        # map first splitting vertex (a, b)<-c to fusion vertex b<-(dual(a), c)
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
            row = indexmap[treeindex_data((fr, fl))]
            @inbounds U[row, col] = conj(factor)
        else
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
                        N₁ > 1 && !isone(fl′.innerlines[1]) && continue
                        coupled = fl′.coupled
                        uncoupled = Base.tail(Base.tail(fl′.uncoupled))
                        isdual = Base.tail(Base.tail(fl′.isdual))
                        inner = N₁ <= 3 ? () : Base.tail(Base.tail(fl′.innerlines))
                        vertices = N₁ <= 2 ? () : Base.tail(Base.tail(fl′.vertices))
                        fl = FusionTree{I}(uncoupled, coupled, isdual, inner, vertices)
                        for (fr, coeff2) in fr_coeffs
                            coeff = factor * coeff1 * conj(coeff2)
                            row = indexmap[treeindex_data((fr, fl))]
                            @inbounds U[row, col] = conj(coeff)
                        end
                    end
                end
            end
        end
    end
    return dst => U
end

# clockwise cyclic permutation while preserving (N₁, N₂): foldright & bendleft
# anticlockwise cyclic permutation while preserving (N₁, N₂): foldleft & bendright
# These are utility functions that preserve the type of the input/output trees,
# and are therefore used to craft type-stable transpose implementations.

function cycleclockwise(src::Union{FusionTreePair, FusionTreeBlock})
    if numout(src) > 0
        tmp, U₁ = foldright(src)
        dst, U₂ = bendleft(tmp)
    else
        tmp, U₁ = bendleft(src)
        dst, U₂ = foldright(tmp)
    end
    return dst => U₂ * U₁
end
function cycleanticlockwise(src::Union{FusionTreePair, FusionTreeBlock})
    if numin(src) > 0
        tmp, U₁ = foldleft(src)
        dst, U₂ = bendright(tmp)
    else
        tmp, U₁ = bendright(src)
        dst, U₂ = foldleft(tmp)
    end
    return dst => U₂ * U₁
end

# COMPOSITE DUALITY MANIPULATIONS PART 1: Repartition and transpose
#-------------------------------------------------------------------
# -> composite manipulations that depend on the duality (rigidity) and pivotal structure
# -> planar manipulations that do not require braiding, everything is in Fsymbol (A/Bsymbol)
# -> transpose expressed as cyclic permutation

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
@inline function repartition(src::Union{FusionTreePair, FusionTreeBlock}, N::Int)
    @assert 0 <= N <= numind(src)
    return repartition(src, Val(N))
end

#=
Using a generated function here to ensure type stability by unrolling the loops:
```julia
dst, U = bendleft/right(src)

# repeat the following 2 lines N - 1 times
dst, Utmp = bendleft/right(dst)
U = Utmp * U

return dst, U
```
=#
function _repartition_body(N)
    if N == 0
        ex = quote
            T = sectorscalartype(sectortype(src))
            if FusionStyle(src) === UniqueFusion()
                return src => one(T)
            else
                U = copyto!(zeros(T, length(src), length(src)), LinearAlgebra.I)
                return src, U
            end
        end
    else
        f = N < 0 ? bendleft : bendright
        ex_rep = Expr(:block)
        for _ in 1:(abs(N) - 1)
            push!(ex_rep.args, :((dst, Utmp) = $f(dst)))
            push!(ex_rep.args, :(U = Utmp * U))
        end
        ex = quote
            dst, U = $f(src)
            $ex_rep
            return dst => U
        end
    end
    return ex
end
@generated function repartition(src::Union{FusionTreePair, FusionTreeBlock}, ::Val{N}) where {N}
    return _repartition_body(numout(src) - N)
end

"""
    transpose((f₁, f₂)::FusionTreePair{I}, p::Index2Tuple{N₁, N₂}) where {I, N₁, N₂}
        -> <:AbstractDict{<:FusionTreePair{I, N₁, N₂}}, <:Number}

Input is a double fusion tree that describes the fusion of a set of incoming uncoupled
sectors to a set of outgoing uncoupled sectors, represented using the individual trees of
outgoing (`t1`) and incoming sectors (`t2`) respectively (with identical coupled sector
`t1.coupled == t2.coupled`). Computes new trees and corresponding coefficients obtained from
repartitioning and permuting the tree such that sectors `p1` become outgoing and sectors
`p2` become incoming.
"""
function Base.transpose(src::Union{FusionTreePair, FusionTreeBlock}, p::Index2Tuple)
    numind(src) == length(p[1]) + length(p[2]) || throw(ArgumentError("invalid permutation p"))
    p′ = linearizepermutation(p..., numout(src), numin(src))
    iscyclicpermutation(p′) || throw(ArgumentError("invalid permutation p"))
    return fstranspose((src, p))
end

const FSPTransposeKey{I, N₁, N₂} = Tuple{FusionTreePair{I}, Index2Tuple{N₁, N₂}}
const FSBTransposeKey{I, N₁, N₂} = Tuple{FusionTreeBlock{I}, Index2Tuple{N₁, N₂}}

Base.@assume_effects :foldable function _fsdicttype(::Type{T}) where {I, N₁, N₂, T <: FSPTransposeKey{I, N₁, N₂}}
    E = sectorscalartype(I)
    return Pair{fusiontreetype(I, N₁, N₂), E}
end
Base.@assume_effects :foldable function _fsdicttype(::Type{T}) where {I, N₁, N₂, T <: FSBTransposeKey{I, N₁, N₂}}
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)
    E = sectorscalartype(I)
    return Pair{FusionTreeBlock{I, N₁, N₂, Tuple{F₁, F₂}}, Matrix{E}}
end

@cached function fstranspose(key::K)::_fsdicttype(K) where {I, N₁, N₂, K <: Union{FSPTransposeKey{I, N₁, N₂}, FSBTransposeKey{I, N₁, N₂}}}
    src, (p1, p2) = key

    N = N₁ + N₂
    p = linearizepermutation(p1, p2, numout(src), numin(src))

    dst, U = repartition(src, N₁)
    length(p) == 0 && return dst => U
    i1 = findfirst(==(1), p)::Int
    i1 == 1 && return dst => U

    Nhalf = N >> 1
    while 1 < i1 ≤ Nhalf
        dst, U_tmp = cycleanticlockwise(dst)
        U = U_tmp * U
        i1 -= 1
    end
    while Nhalf < i1
        dst, U_tmp = cycleclockwise(dst)
        U = U_tmp * U
        i1 = mod1(i1 + 1, N)
    end

    return dst => U
end

CacheStyle(::typeof(fstranspose), k::FSPTransposeKey{I}) where {I} =
    FusionStyle(I) isa UniqueFusion ? NoCache() : GlobalLRUCache()
CacheStyle(::typeof(fstranspose), k::FSBTransposeKey{I}) where {I} =
    FusionStyle(I) isa UniqueFusion ? NoCache() : GlobalLRUCache()

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
    if FusionStyle(I) isa UniqueFusion
        (f₁′, f₂′), coeff′ = repartition((f₁, f₂), N)
        for (f₁′′, coeff′′) in planar_trace(f₁′, (q1′, q2′))
            (f12′′′, coeff′′′) = transpose((f₁′′, f₂′), (p1′, p2′))
            coeff = coeff′ * coeff′′ * coeff′′′
            iszero(coeff) || (newtrees[f12′′′] = get(newtrees, f12′′′, zero(coeff)) + coeff)
        end
    else
        # TODO: this is a bit of a hack to fix the traces for now
        src = FusionTreeBlock([(f₁, f₂)])
        dst, U = repartition(src, N)
        for ((f₁′, f₂′), coeff′) in zip(fusiontrees(dst), U)
            for (f₁′′, coeff′′) in planar_trace(f₁′, (q1′, q2′))
                src′ = FusionTreeBlock([(f₁′′, f₂′)])
                dst′, U′ = transpose(src′, (p1′, p2′))
                for (f12′′′, coeff′′′) in zip(fusiontrees(dst′), U′)
                    coeff = coeff′ * coeff′′ * coeff′′′
                    iszero(coeff) || (newtrees[f12′′′] = get(newtrees, f12′′′, zero(coeff)) + coeff)
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
