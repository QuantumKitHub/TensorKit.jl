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

# generate the relevant fusion tree pair after the action of bendright,
# but with a default vertex label of ν = 1 in the case of multiplicities
function _bendright_treepair((f₁, f₂)::FusionTreePair)
    I = sectortype((f₁, f₂))
    N₁, N₂ = numout((f₁, f₂)), numin((f₁, f₂))
    c = f₁.coupled
    a = N₁ == 1 ? leftunit(f₁.uncoupled[1]) : (N₁ == 2 ? f₁.uncoupled[1] : f₁.innerlines[end])
    b = f₁.uncoupled[N₁]

    # construct the new fusiontree pair
    uncoupled₁ = TupleTools.front(f₁.uncoupled)
    isdual₁ = TupleTools.front(f₁.isdual)
    inner₁ = N₁ > 2 ? TupleTools.front(f₁.innerlines) : ()
    vertices₁ = N₁ > 1 ? TupleTools.front(f₁.vertices) : ()
    f₁′ = FusionTree{I}(uncoupled₁, a, isdual₁, inner₁, vertices₁)

    uncoupled₂ = (f₂.uncoupled..., dual(b))
    isdual₂ = (f₂.isdual..., !(f₁.isdual[N₁]))
    inner₂ = N₂ > 1 ? (f₂.innerlines..., c) : ()
    vertices₂ = N₂ > 0 ? (f₂.vertices..., 1) : ()
    f₂′ = FusionTree{I}(uncoupled₂, a, isdual₂, inner₂, vertices₂)

    return (a, b, c), (f₁′, f₂′)
end

# also for a final splitting vertex without multiplicity (`only` throws otherwise)
function bendright((f₁, f₂)::FusionTreePair)
    N₁ = numout((f₁, f₂))
    (a, b, c), (f₁′, f₂′) = _bendright_treepair((f₁, f₂))

    # compute the coefficient
    coeff₀ = sqrtdim(c) * invsqrtdim(a)
    f₁.isdual[N₁] && (coeff₀ *= conj(frobenius_schur_phase(dual(b))))
    coeff = coeff₀ * only(Bsymbol(a, b, c))

    return (f₁′, f₂′) => coeff
end
function bendright(src::FusionTreeBlock)
    I = sectortype(src)
    N₁ = numout(src)
    N₂ = numin(src)
    @assert N₁ > 0

    rows = TreeRows(fusiontreetype(I, N₁ - 1, N₂ + 1), length(src))
    U = zeros(fusionscalartype(I), length(src), length(src))

    for (col, (f₁, f₂)) in enumerate(fusiontrees(src))
        (a, b, c), (f₁′, f₂′) = _bendright_treepair((f₁, f₂))
        coeff₀ = sqrtdim(c) * invsqrtdim(a)
        if f₁.isdual[N₁]
            coeff₀ *= conj(frobenius_schur_phase(dual(b)))
        end
        if FusionStyle(I) isa MultiplicityFreeFusion
            coeff = coeff₀ * Bsymbol(a, b, c)
            row = row!(rows, (f₁′, f₂′))
            @inbounds U[row, col] = coeff
        else
            Bmat = Bsymbol(a, b, c)
            μ = N₁ > 1 ? f₁.vertices[end] : 1
            uncoupled₂ = f₂′.uncoupled
            coupled₂ = f₂′.coupled
            isdual₂ = f₂′.isdual
            inner₂ = f₂′.innerlines
            for ν in axes(Bmat, 2)
                coeff = coeff₀ * Bmat[μ, ν]
                iszero(coeff) && continue
                vertices₂ = N₂ > 0 ? (f₂.vertices..., ν) : ()
                f₂′ = FusionTree(uncoupled₂, coupled₂, isdual₂, inner₂, vertices₂)
                row = row!(rows, (f₁′, f₂′))
                @inbounds U[row, col] = coeff
            end
        end
    end
    return FusionTreeBlock(rows.trees) => U
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
    (f₂′, f₁′), coeff = bendright((f₂, f₁))
    return (f₁′, f₂′) => conj(coeff)
end

# !! note that this is more or less a copy of bendright through
# (f1, f2) => conj(coeff) for ((f2, f1), coeff) in bendleft(src)
function bendleft(src::FusionTreeBlock)
    I = sectortype(src)
    N₁ = numout(src)
    N₂ = numin(src)
    @assert N₂ > 0

    rows = TreeRows(fusiontreetype(I, N₁ + 1, N₂ - 1), length(src))
    U = zeros(fusionscalartype(I), length(src), length(src))

    for (col, (f₁, f₂)) in enumerate(fusiontrees(src))
        (a, b, c), (f₂′, f₁′) = _bendright_treepair((f₂, f₁))
        coeff₀ = sqrtdim(c) * invsqrtdim(a)
        if f₂.isdual[N₂]
            coeff₀ *= conj(frobenius_schur_phase(dual(b)))
        end
        if FusionStyle(I) isa MultiplicityFreeFusion
            coeff = coeff₀ * Bsymbol(a, b, c)
            row = row!(rows, (f₁′, f₂′))
            @inbounds U[row, col] = conj(coeff)
        else
            Bmat = Bsymbol(a, b, c)
            μ = N₂ > 1 ? f₂.vertices[end] : 1
            uncoupled₁ = f₁′.uncoupled
            coupled₁ = f₁′.coupled
            isdual₁ = f₁′.isdual
            inner₁ = f₁′.innerlines
            for ν in axes(Bmat, 2)
                coeff = coeff₀ * Bmat[μ, ν]
                iszero(coeff) && continue
                vertices₁ = N₁ > 0 ? (f₁.vertices..., ν) : ()
                f₁′ = FusionTree(uncoupled₁, coupled₁, isdual₁, inner₁, vertices₁)
                row = row!(rows, (f₁′, f₂′))
                @inbounds U[row, col] = conj(coeff)
            end
        end
    end
    return FusionTreeBlock(rows.trees) => U
end

@doc """
    foldright((f₁, f₂)::FusionTreePair) -> (f₃, f₄) => coeff
    foldright(src::FusionTreeBlock) -> dst => coeffs

Map the first splitting vertex `a ⊗ b ← c` of `src` to a fusion vertex `b ← dual(a) ⊗ c`,
and reexpress as a linear combination of standard basis trees.
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

    a = f₁.uncoupled[1]
    κₐ = frobenius_schur_phase(a)
    isduala = f₁.isdual[1]
    f₁′, coeff₁ = map(only, multi_Fmove(f₁))
    b = f₁′.coupled
    c = f₁.coupled

    f₂′, coeff₂ = map(only, multi_Fmove_inv(dual(a), b, f₂, !isduala))
    coeff = sqrtdim(f₁.coupled) * invsqrtdim(b) * coeff₁ * Asymbol(a, b, c) * conj(coeff₂)

    return (f₁′, f₂′) => (isduala ? coeff * κₐ : coeff)
end

function foldright(src::FusionTreeBlock)
    I = sectortype(src)
    N₁ = numout(src)
    @assert N₁ > 0

    rows = TreeRows(fusiontreetype(I, N₁ - 1, numin(src) + 1), length(src))

    f₁, f₂ = first(fusiontrees(src))
    a::I = f₁.uncoupled[1]
    κₐ = frobenius_schur_phase(a)
    isduala = f₁.isdual[1]

    cache₁ = Dict(f₁ => multi_Fmove(f₁))
    f₁′, coeff₁ = first.(cache₁[f₁])
    b::I = f₁′.coupled
    cache₂ = Dict((b, f₂) => multi_Fmove_inv(dual(a), b, f₂, !isduala))
    c::I = f₁.coupled
    cache₃ = Dict((b, c) => Asymbol(a, b, c))

    U = zeros(eltype(coeff₁), length(src), length(src))
    for (col, (f₁, f₂)) in enumerate(fusiontrees(src))
        f₁′s, coeffs₁ = get!(cache₁, f₁) do
            multi_Fmove(f₁)
        end
        for (f₁′, coeff₁) in zip(f₁′s, coeffs₁)
            b = f₁′.coupled
            c = f₁.coupled
            A = get!(cache₃, (b, c)) do
                Asymbol(a, b, c)
            end
            f₂′s, coeffs₂ = get!(cache₂, (b, f₂)) do
                multi_Fmove_inv(dual(a), b, f₂, !isduala)
            end
            coeff₀ = sqrtdim(c) * invsqrtdim(b)
            for (f₂′, coeff₂) in zip(f₂′s, coeffs₂)
                coeff = coeff₀ * (coeff₂' * (transpose(A) * coeff₁))
                if isduala
                    coeff *= κₐ
                end
                row = row!(rows, (f₁′, f₂′))
                @inbounds U[row, col] += coeff
            end
        end
    end
    return FusionTreeBlock(rows.trees) => U
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

function foldleft(src::FusionTreeBlock)
    I = sectortype(src)
    N₁ = numin(src)
    N₂ = numout(src)
    @assert N₁ > 0

    rows = TreeRows(fusiontreetype(I, N₂ + 1, N₁ - 1), length(src))

    f₁, f₂ = first(fusiontrees(src))
    a::I = f₂.uncoupled[1]
    κₐ = frobenius_schur_phase(a)
    isduala = f₂.isdual[1]

    cache₂ = Dict(f₂ => multi_Fmove(f₂))
    f₂′, coeff₂ = first.(cache₂[f₂])
    b::I = f₂′.coupled
    cache₁ = Dict((b, f₁) => multi_Fmove_inv(dual(a), b, f₁, !isduala))
    c::I = f₂.coupled
    cache₃ = Dict((b, c) => Asymbol(a, b, c))

    U = zeros(eltype(coeff₂), length(src), length(src))
    for (col, (f₁, f₂)) in enumerate(fusiontrees(src))
        f₂′s, coeffs₂ = get!(cache₂, f₂) do
            multi_Fmove(f₂)
        end
        for (f₂′, coeff₂) in zip(f₂′s, coeffs₂)
            b = f₂′.coupled
            c = f₂.coupled
            A = get!(cache₃, (b, c)) do
                Asymbol(a, b, c)
            end
            f₁′s, coeffs₁ = get!(cache₁, (b, f₁)) do
                multi_Fmove_inv(dual(a), b, f₁, !isduala)
            end
            coeff₀ = sqrtdim(c) * invsqrtdim(b)
            for (f₁′, coeff₁) in zip(f₁′s, coeffs₁)
                coeff = coeff₀ * conj(coeff₁' * (transpose(A) * coeff₂))
                if isduala
                    coeff *= conj(κₐ)
                end
                row = row!(rows, (f₁′, f₂′))
                @inbounds U[row, col] += coeff
            end
        end
    end
    return FusionTreeBlock(rows.trees) => U
end

# clockwise cyclic permutation while preserving (N₁, N₂): foldright & bendleft
# anticlockwise cyclic permutation while preserving (N₁, N₂): foldleft & bendright
# These are utility functions that preserve the type of the input/output trees,
# and are therefore used to craft type-stable transpose implementations.

@doc """
    cycleclockwise((f₁, f₂)::FusionTreePair) -> (f₃, f₄) => coeff
    cycleclockwise(src::FusionTreeBlock) -> dst => coeffs

Bend the last fusion sector to the splitting side, and fold the first splitting sector to the fusion side.
```
    | ╰─┬─╯ |  ╭──╮     ╰─┬─╯ |   |
    |   ╰─┬─╯  |  |       ╰─┬─╯   |
    |     ╰ ⋯ ┬╯  |         ╰ ⋯ ┬─╯
    |         |   |  →          |
    |     ╭ ⋯ ┴╮  |         ╭ ⋯ ┴─╮
    |   ╭─┴─╮  |  |       ╭─┴─╮   |
    ╰───┴─╮ |  |  |     ╭─┴─╮ |   |
```

See also [`cycleanticlockwise`](@ref).
""" cycleclockwise

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

@doc """
    cycleanticlockwise((f₁, f₂)::FusionTreePair) -> (f₃, f₄) => coeff
    cycleanticlockwise(src::FusionTreeBlock) -> dst => coeffs

Bend the last splitting sector to the fusion side, and fold the first fusion sector to the splitting side.
```
    ╭──╮   |  |  |     ╰─┬─╯ |   |
    |  ╰─┬─╯  |  |       ╰─┬─╯   |
    |    ╰ ⋯ ┬╯  |         ╰ ⋯ ┬─╯
    |        |   |  →          |
    |    ╭ ⋯ ┴╮  |         ╭ ⋯ ┴─╮
    |  ╭─┴─╮  |  |       ╭─┴─╮   |
    |  |   |  ╰──╯     ╭─┴─╮ |   |
```

See also [`cycleanticlockwise`](@ref).
""" cycleanticlockwise


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
# NOTE: `_repartition_body` must be defined *before* the `@generated repartition` that calls
# it, so that it is visible to the generator's world age. Otherwise a precompile workload that
# first triggers `repartition` fails with `UndefVarError: _repartition_body` (at runtime the
# first call happens at a later world age, which masks the ordering issue).
function _repartition_body(N)
    if N == 0
        ex = quote
            T = fusionscalartype(sectortype(src))
            if src isa FusionTreePair
                return src => one(T)
            else
                U = copyto!(zeros(T, length(src), length(src)), LinearAlgebra.I)
                return src => U
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
    N = numind(src)
    N == length(p[1]) + length(p[2]) || throw(ArgumentError("invalid permutation p = $p of length N = $N"))
    p′ = linearizepermutation(p..., numout(src), numin(src))
    iscyclicpermutation(p′) || throw(ArgumentError("invalid cyclic or planar permutation p = $p"))
    return fstranspose((src, p))
end

const FSPTransposeKey{I, N₁, N₂} = Tuple{FusionTreePair{I}, Index2Tuple{N₁, N₂}}
const FSBTransposeKey{I, N₁, N₂} = Tuple{FusionTreeBlock{I}, Index2Tuple{N₁, N₂}}

Base.@assume_effects :foldable function _fsdicttype(::Type{T}) where {I, N₁, N₂, T <: FSPTransposeKey{I, N₁, N₂}}
    E = fusionscalartype(I)
    return Pair{fusiontreetype(I, N₁, N₂), E}
end
Base.@assume_effects :foldable function _fsdicttype(::Type{T}) where {I, N₁, N₂, T <: FSBTransposeKey{I, N₁, N₂}}
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)
    E = fusionscalartype(I)
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

# leg positions of a planar trace of a tree pair with `N₁` outgoing and `N₂` incoming legs, after
# repartitioning all legs to the outgoing side: the traced pairs `q′`, the remaining legs `r` in
# increasing order and the permutation `p′` that is applied after removing the traced legs
function _planar_trace_indices(::Val{N₁}, ::Val{N₂}, (p₁, p₂)::Index2Tuple, (q₁, q₂)::Index2Tuple) where {N₁, N₂}
    length(q₁) == length(q₂) ||
        throw(ArgumentError(lazy"trace index tuples q₁ and q₂ must have equal length, got $(length(q₁)) and $(length(q₂))"))
    N = length(p₁) + length(p₂) + 2 * length(q₁)
    N₁ + N₂ == N ||
        throw(ArgumentError(lazy"fusion tree pair has $(N₁ + N₂) indices, but permutation expects $N = $(length(p₁)) + $(length(p₂)) + 2×$(length(q₁))"))
    linearindex = (ntuple(identity, Val(N₁))..., reverse(N₁ .+ ntuple(identity, Val(N₂)))...)
    q₁′ = TupleTools.getindices(linearindex, q₁)
    q₂′ = TupleTools.getindices(linearindex, q₂)
    q′ = (q₁′..., q₂′...)
    p₁′ = map(l -> l - count(l .> q′), TupleTools.getindices(linearindex, p₁))
    p₂′ = map(l -> l - count(l .> q′), TupleTools.getindices(linearindex, p₂))
    r = TupleTools.sort(TupleTools.getindices(linearindex, (p₁..., p₂...)))
    return (p₁′, p₂′), (q₁′, q₂′), r
end

# the sectors of the traced legs have to be dual to each other for a nonzero trace, where the
# sector of an incoming leg is the dual of the corresponding uncoupled sector of the fusion tree
function trace_sectors_match(src::Union{FusionTreePair, FusionTreeBlock}, (q₁, q₂)::Index2Tuple)
    f₁, f₂ = src isa FusionTreePair ? src : first(fusiontrees(src))
    N₁ = length(f₁)
    legsector(i) = i <= N₁ ? f₁.uncoupled[i] : dual(f₂.uncoupled[i - N₁])
    return all(map((i, j) -> legsector(i) == dual(legsector(j)), q₁, q₂))
end

"""
    planar_trace((f₁, f₂)::FusionTreePair, (p₁, p₂)::Index2Tuple, (q₁, q₂)::Index2Tuple) -> (f₁′, f₂′) => coeff
    planar_trace(src::FusionTreeBlock, (p₁, p₂)::Index2Tuple, (q₁, q₂)::Index2Tuple) -> dst => U

Perform a planar trace of the legs `q₁[i]` with `q₂[i]` of a fusion tree pair, followed by a
planar permutation `(p₁, p₂)` of the remaining legs. The sectors of the traced legs have to be
dual to each other.

For `FusionStyle(I) isa UniqueFusion`, the input is a single fusion tree pair, which is mapped
to a single output tree pair with a coefficient. For generic `FusionStyle`s, the input and
output consist of `FusionTreeBlock`s, where `src` and `dst` contain all tree pairs of the
traced and remaining legs respectively, and `U[j, i]` is the coefficient of the `j`th tree pair
of `dst` in the trace of the `i`th tree pair of `src`.
"""
function planar_trace(src::Union{FusionTreePair, FusionTreeBlock}, (p₁, p₂)::Index2Tuple, (q₁, q₂)::Index2Tuple)
    src isa FusionTreePair && @assert FusionStyle(src) === UniqueFusion()
    p′, q′, r = _planar_trace_indices(Val(numout(src)), Val(numin(src)), (p₁, p₂), (q₁, q₂))
    trace_sectors_match(src, (q₁, q₂)) ||
        throw(ArgumentError(lazy"sectors of the traced legs $q₁ and $q₂ are not dual to each other"))
    isempty(q₁) && return transpose(src, (p₁, p₂))

    # bend all legs to the outgoing side, where the trace only acts on the splitting trees
    src′, U₁ = repartition(src, numind(src))
    traced, U₂ = _planar_trace_splitting(src′, q′, r)
    # for multiple fusion channels, the remaining legs need not couple to the unit
    if traced isa FusionTreeBlock && isempty(traced)
        I = sectortype(src)
        F = fusiontreetype(I, length(p₁), length(p₂))
        return FusionTreeBlock(F[]) => zeros(fusionscalartype(I), 0, length(src))
    else
        dst, U₃ = transpose(traced, p′)
        return dst => U₃ * U₂ * U₁
    end
end

function _planar_trace_splitting((f, f₀)::FusionTreePair, q′, r)
    f′, coeff = only(planar_trace(f, q′))
    return (f′, f₀) => coeff
end
function _planar_trace_splitting(src::FusionTreeBlock, q′, r)
    I = sectortype(src)
    f₀, = first(fusiontrees(src))
    uncoupled = TupleTools.getindices(f₀.uncoupled, r)
    isdual = TupleTools.getindices(f₀.isdual, r)
    dst = FusionTreeBlock{I}((uncoupled, ()), (isdual, ()))
    U = zeros(fusionscalartype(I), length(dst), length(src))
    positions = Dict(f => i for (i, (f, _)) in enumerate(fusiontrees(dst)))
    for (j, (f, _)) in enumerate(fusiontrees(src)), (f′, coeff) in planar_trace(f, q′)
        U[positions[f′], j] += coeff
    end
    return dst => U
end

"""
    planar_trace(f::FusionTree, (q₁, q₂)::Index2Tuple)
        -> <:AbstractDict{<:FusionTree, <:Number}

Perform a planar trace of the uncoupled indices of the fusion tree `f` at `q₁` with those at `q₂`,
where `q₁[i]` is connected to `q₂[i]` for all `i`. The result is returned as a dictionary of output
trees and corresponding coefficients.
"""
function planar_trace(f::FusionTree, (q₁, q₂)::Index2Tuple)
    length(q₁) == length(q₂) ||
        throw(ArgumentError(lazy"trace index tuples q₁ and q₂ must have equal length, got $(length(q₁)) and $(length(q₂))"))
    I = sectortype(f)
    T = fusionscalartype(I)
    F = fusiontreetype(I, length(f) - 2 * length(q₁))
    newtrees = FusionTreeDict{F, T}()
    isempty(q₁) && return push!(newtrees, f => one(T))

    for (i, j) in zip(q₁, q₂)
        (f.uncoupled[i] == dual(f.uncoupled[j]) && f.isdual[i] != f.isdual[j]) ||
            return newtrees
    end
    # Planar traces are over neighboring indices, but might be nested, so that
    # index i can be traced with i+3, if index i+1 is also traced with index i+2.
    # We thus handle the total trace recursively, by first looking for and
    # tracing away neighbouring pairs.
    k = 1
    i = j = 0
    while k <= length(q₁)
        if mod1(q₁[k] + 1, length(f)) == q₂[k]
            i = q₁[k]
            j = q₂[k]
            break
        elseif mod1(q₂[k] + 1, length(f)) == q₁[k]
            i = q₂[k]
            j = q₁[k]
            break
        else
            k += 1
        end
    end
    k > length(q₁) &&
        throw(ArgumentError(lazy"indices $q₁ and $q₂ do not form a valid planar trace on a fusion tree with $(length(f)) legs: no neighboring pair found among the remaining trace indices"))

    q₁′ = let i = i, j = j
        map(l -> (l - (l > i) - (l > j)), TupleTools.deleteat(q₁, k))
    end
    q₂′ = let i = i, j = j
        map(l -> (l - (l > i) - (l > j)), TupleTools.deleteat(q₂, k))
    end
    for (f′, coeff′) in elementary_trace(f, i)
        for (f′′, coeff′′) in planar_trace(f′, (q₁′, q₂′))
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
    elementary_trace(f::FusionTree{I, N}, i) where {I,N} -> <:AbstractDict{FusionTree{I,N-2}, <:Number}

Perform an elementary trace of neighbouring uncoupled indices `i` and
`i+1` on a fusion tree `f`, and returns the result as a dictionary of output trees and
corresponding coefficients.
"""
function elementary_trace(f::FusionTree{I, N}, i) where {I, N}
    (N > 1 && 1 <= i <= N) ||
        throw(ArgumentError("Cannot trace outputs i=$i and i+1 out of only $N outputs"))
    i < N || isunit(f.coupled) ||
        throw(ArgumentError("Cannot trace outputs i=$N and 1 of fusion tree that couples to non-trivial sector"))

    T = fusionscalartype(I)
    F = fusiontreetype(I, N - 2)
    newtrees = FusionTreeDict{F, T}()

    j = mod1(i + 1, N)
    b = f.uncoupled[i]
    b′ = f.uncoupled[j]
    # if trace is zero, return empty dict
    (b == dual(b′) && f.isdual[i] != f.isdual[j]) || return newtrees
    if i < N
        fleft, fremainder = split(f, i - 1)
        ftrace, fright = split(fremainder, 3)
        a = ftrace.uncoupled[1] # == fleft.coupled
        d = ftrace.coupled # == fright.uncoupled[1]
        a == d || return newtrees
        f′ = join(fleft, fright)
        coeff = sqrtdim(b)
        if i > 1
            c = ftrace.innerlines[1]
            μ, ν = ftrace.vertices
            coeff *= Fsymbol(a, b, dual(b), a, c, rightunit(a))[μ, ν, 1, 1]
        end
        if ftrace.isdual[2]
            coeff *= frobenius_schur_phase(b)
        end
        push!(newtrees, f′ => coeff)
        return newtrees
    else # i == N
        fleft, fremainder = split(f, N - 1)
        trees, coeffvecs = multi_Fmove(fleft)
        for (f′, coeffvec) in zip(trees, coeffvecs)
            isunit(f′.coupled) || continue
            coeff = only(coeffvec)
            coeff *= sqrtdim(b)
            if !(f.isdual[N])
                coeff *= conj(frobenius_schur_phase(b))
            end
            push!(newtrees, f′ => coeff)
        end
        return newtrees
    end
end
