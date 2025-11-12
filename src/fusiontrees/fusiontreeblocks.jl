struct FusionTreeBlock{I, N₁, N₂, F <: FusionTreePair{I, N₁, N₂}}
    trees::Vector{F}
end

function FusionTreeBlock{I}(
        uncoupled::Tuple{NTuple{N₁, I}, NTuple{N₂, I}},
        isdual::Tuple{NTuple{N₁, Bool}, NTuple{N₂, Bool}};
        sizehint::Int = 0
    ) where {I <: Sector, N₁, N₂}
    F = fusiontreetype(I, N₁, N₂)
    trees = Vector{F}(undef, 0)
    sizehint > 0 && sizehint!(trees, sizehint)

    if N₁ == N₂ == 0
        return FusionTreeBlock(trees)
    elseif N₁ == 0
        cs = sort!(collect(filter(isone, ⊗(uncoupled[2]...))))
    elseif N₂ == 0
        cs = sort!(collect(filter(isone, ⊗(uncoupled[1]...))))
    else
        cs = sort!(collect(intersect(⊗(uncoupled[1]...), ⊗(uncoupled[2]...))))
    end

    for c in cs
        for f₁ in fusiontrees(uncoupled[1], c, isdual[1]),
                f₂ in fusiontrees(uncoupled[2], c, isdual[2])

            push!(trees, (f₁, f₂))
        end
    end
    return FusionTreeBlock(trees)
end

Base.@constprop :aggressive function Base.getproperty(block::FusionTreeBlock, prop::Symbol)
    if prop === :uncoupled
        f₁, f₂ = first(block.trees)
        return f₁.uncoupled, f₂.uncoupled
    elseif prop === :isdual
        f₁, f₂ = first(block.trees)
        return f₁.isdual, f₂.isdual
    else
        return getfield(block, prop)
    end
end

Base.propertynames(::FusionTreeBlock, private::Bool = false) = (:trees, :uncoupled, :isdual)

sectortype(::Type{<:FusionTreeBlock{I}}) where {I} = I
numout(fs::FusionTreeBlock) = numout(typeof(fs))
numout(::Type{<:FusionTreeBlock{I, N₁}}) where {I, N₁} = N₁
numin(fs::FusionTreeBlock) = numin(typeof(fs))
numin(::Type{<:FusionTreeBlock{I, N₁, N₂}}) where {I, N₁, N₂} = N₂
numind(fs::FusionTreeBlock) = numind(typeof(fs))
numind(::Type{T}) where {T <: FusionTreeBlock} = numin(T) + numout(T)

fusiontrees(block::FusionTreeBlock) = block.trees
Base.length(block::FusionTreeBlock) = length(fusiontrees(block))

# Within one block, all values of uncoupled and isdual are equal, so avoid hashing these
function treeindex_data((f₁, f₂))
    I = sectortype(f₁)
    if FusionStyle(I) isa GenericFusion
        return (f₁.coupled, f₁.innerlines..., f₂.innerlines...),
            (f₁.vertices..., f₂.vertices...)
    elseif FusionStyle(I) isa MultipleFusion
        return (f₁.coupled, f₁.innerlines..., f₂.innerlines...)
    else # there should be only a single element anyways
        return false
    end
end
function treeindex_map(fs::FusionTreeBlock)
    I = sectortype(fs)
    return fusiontreedict(I)(treeindex_data(f) => ind for (ind, f) in enumerate(fusiontrees(fs)))
end

# Manipulations
# -------------
function transformation_matrix(transform, dst::FusionTreeBlock{I}, src::FusionTreeBlock{I}) where {I}
    U = zeros(sectorscalartype(I), length(dst), length(src))
    indexmap = treeindex_map(dst)
    for (col, f) in enumerate(fusiontrees(src))
        for (f′, c) in transform(f)
            row = indexmap[f′]
            U[row, col] = c
        end
    end
    return U
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
            coeff₀ *= conj(frobeniusschur(dual(b)))
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
                row = indexmap[(f₁′, f₂′)]
                @inbounds U[row, col] = coeff
            end
        end
    end

    return dst, U
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
            coeff₀ *= conj(frobeniusschur(dual(b)))
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

    return dst, U
end

function foldright(src::FusionTreeBlock)
    uncoupled_dst = (
        Base.tail(src.uncoupled[1]),
        (dual(first(src.uncoupled[1])), src.uncoupled[2]...),
    )
    isdual_dst = (Base.tail(src.isdual[1]), (!first(src.isdual[1]), src.isdual[2]...))
    I = sectortype(src)
    N₁ = numout(src)
    N₂ = numin(src)
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
            factor *= conj(frobeniusschur(a))
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

    return dst, U
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
            factor *= conj(frobeniusschur(a))
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
    return dst, U
end

function cycleclockwise(src::FusionTreeBlock)
    if numout(src) > 0
        tmp, U₁ = foldright(src)
        dst, U₂ = bendleft(tmp)
    else
        tmp, U₁ = bendleft(src)
        dst, U₂ = foldright(tmp)
    end
    return dst, U₂ * U₁
end

function cycleanticlockwise(src::FusionTreeBlock)
    if numin(src) > 0
        tmp, U₁ = foldleft(src)
        dst, U₂ = bendright(tmp)
    else
        tmp, U₁ = bendright(src)
        dst, U₂ = foldleft(tmp)
    end
    return dst, U₂ * U₁
end

@inline function repartition(src::FusionTreeBlock, N::Int)
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
@generated function repartition(src::FusionTreeBlock, ::Val{N}) where {N}
    return _repartition_body(numout(src) - N)
end
function _repartition_body(N)
    if N == 0
        ex = quote
            T = sectorscalartype(sectortype(src))
            U = copyto!(zeros(T, length(src), length(src)), LinearAlgebra.I)
            return src, U
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
            return dst, U
        end
    end
    return ex
end

function Base.transpose(src::FusionTreeBlock, p::Index2Tuple{N₁, N₂}) where {N₁, N₂}
    N = N₁ + N₂
    @assert numind(src) == N
    p′ = linearizepermutation(p..., numout(src), numin(src))
    @assert iscyclicpermutation(p′)
    return _fstranspose((src, p))
end

const _FSTransposeKey{I, N₁, N₂} = Tuple{<:FusionTreeBlock{I}, Index2Tuple{N₁, N₂}}

@cached function _fstranspose(
        key::_FSTransposeKey{I, N₁, N₂}
    )::Tuple{FusionTreeBlock{I, N₁, N₂}, Matrix{sectorscalartype(I)}} where {I, N₁, N₂}
    src, (p1, p2) = key

    N = N₁ + N₂
    p = linearizepermutation(p1, p2, numout(src), numin(src))

    dst, U = repartition(src, N₁)
    length(p) == 0 && return dst, U
    i1 = findfirst(==(1), p)::Int
    i1 == 1 && return dst, U

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

    return dst, U
end

function CacheStyle(::typeof(_fstranspose), k::_FSTransposeKey{I}) where {I}
    if FusionStyle(I) == UniqueFusion()
        return NoCache()
    else
        return GlobalLRUCache()
    end
end

function artin_braid(src::FusionTreeBlock{I, N, 0}, i; inv::Bool = false) where {I, N}
    1 <= i < N ||
        throw(ArgumentError("Cannot swap outputs i=$i and i+1 out of only $N outputs"))

    uncoupled = src.uncoupled[1]
    a, b = uncoupled[i], uncoupled[i + 1]
    uncoupled′ = TupleTools.setindex(uncoupled, b, i)
    uncoupled′ = TupleTools.setindex(uncoupled′, a, i + 1)
    coupled′ = rightunit(src.uncoupled[1][N])

    isdual = src.isdual[1]
    isdual′ = TupleTools.setindex(isdual, isdual[i], i + 1)
    isdual′ = TupleTools.setindex(isdual′, isdual[i + 1], i)
    dst = FusionTreeBlock{I}((uncoupled′, ()), (isdual′, ()); sizehint = length(src))

    oneT = one(sectorscalartype(I))

    indexmap = treeindex_map(dst)
    U = zeros(sectorscalartype(I), length(dst), length(src))

    if isone(a) || isone(b) # braiding with trivial sector: simple and always possible
        for (col, (f, f₂)) in enumerate(fusiontrees(src))
            inner = f.innerlines
            inner_extended = (uncoupled[1], inner..., coupled′)
            vertices = f.vertices
            inner′ = inner
            vertices′ = vertices
            if i > 1 # we also need to alter innerlines and vertices
                inner′ = TupleTools.setindex(
                    inner, inner_extended[isone(a) ? (i + 1) : (i - 1)], i - 1
                )
                vertices′ = TupleTools.setindex(vertices′, vertices[i], i - 1)
                vertices′ = TupleTools.setindex(vertices′, vertices[i - 1], i)
            end
            f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner′, vertices′)
            row = indexmap[treeindex_data((f′, f₂))]
            @inbounds U[row, col] = oneT
        end
        return dst, U
    end

    BraidingStyle(I) isa NoBraiding &&
        throw(SectorMismatch(lazy"Cannot braid sectors $a and $b"))

    for (col, (f, f₂)) in enumerate(fusiontrees(src))
        inner = f.innerlines
        inner_extended = (uncoupled[1], inner..., coupled′)
        vertices = f.vertices

        if i == 1
            c = N > 2 ? inner[1] : coupled′
            if FusionStyle(I) isa MultiplicityFreeFusion
                R = oftype(oneT, (inv ? conj(Rsymbol(b, a, c)) : Rsymbol(a, b, c)))
                f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner, vertices)
                row = indexmap[treeindex_data((f′, f₂))]
                @inbounds U[row, col] = R
            else # GenericFusion
                μ = vertices[1]
                Rmat = inv ? Rsymbol(b, a, c)' : Rsymbol(a, b, c)
                for ν in axes(Rmat, 2)
                    R = oftype(oneT, Rmat[μ, ν])
                    iszero(R) && continue
                    vertices′ = TupleTools.setindex(vertices, ν, 1)
                    f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner, vertices′)
                    row = indexmap[treeindex_data((f′, f₂))]
                    @inbounds U[row, col] = R
                end
            end
            continue
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
            row = indexmap[treeindex_data((f′, f₂))]
            @inbounds U[row, col] = coeff
        elseif FusionStyle(I) isa SimpleFusion
            cs = collect(I, intersect(a ⊗ d, e ⊗ conj(b)))
            for c′ in cs
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
                row = indexmap[treeindex_data((f′, f₂))]
                @inbounds U[row, col] = coeff
            end
        else # GenericFusion
            cs = collect(I, intersect(a ⊗ d, e ⊗ conj(b)))
            for c′ in cs
                Rmat1 = inv ? Rsymbol(d, c, e)' : Rsymbol(c, d, e)
                Rmat2 = inv ? Rsymbol(d, a, c′)' : Rsymbol(a, d, c′)
                Fmat = Fsymbol(d, a, b, e, c′, c)
                μ = vertices[i - 1]
                ν = vertices[i]
                for σ in 1:Nsymbol(a, d, c′)
                    for λ in 1:Nsymbol(c′, b, e)
                        coeff = zero(oneT)
                        for ρ in 1:Nsymbol(d, c, e), κ in 1:Nsymbol(d, a, c′)
                            coeff += Rmat1[ν, ρ] * conj(Fmat[κ, λ, μ, ρ]) *
                                conj(Rmat2[σ, κ])
                        end
                        iszero(coeff) && continue
                        vertices′ = TupleTools.setindex(vertices, σ, i - 1)
                        vertices′ = TupleTools.setindex(vertices′, λ, i)
                        inner′ = TupleTools.setindex(inner, c′, i - 1)
                        f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner′, vertices′)
                        row = indexmap[treeindex_data((f′, f₂))]
                        @inbounds U[row, col] = coeff
                    end
                end
            end
        end
    end

    return dst, U
end

function braid(src::FusionTreeBlock{I, N, 0}, p::NTuple{N, Int}, levels::NTuple{N, Int}) where {I, N}
    TupleTools.isperm(p) || throw(ArgumentError("not a valid permutation: $p"))

    if FusionStyle(I) isa UniqueFusion && BraidingStyle(I) isa SymmetricBraiding
        uncoupled′ = TupleTools._permute(src.uncoupled[1], p)
        isdual′ = TupleTools._permute(src.isdual[1], p)
        dst = FusionTreeBlock{I}(uncoupled′, isdual′; sizehint = length(src))
        U = transformation_matrix(dst, src) do (f₁, f₂)
            return ((f₁′, f₂) => c for (f₁′, c) in braid(f₁, p, levels))
        end
    else
        dst, U = repartition(src, N) # TODO: can we avoid this?
        for s in permutation2swaps(p)
            inv = levels[s] > levels[s + 1]
            dst, U_tmp = artin_braid(dst, s; inv)
            U = U_tmp * U
        end
    end
    return dst, U
end

function braid(src::FusionTreeBlock{I}, p::Index2Tuple{N₁, N₂}, levels::Index2Tuple) where {I, N₁, N₂}
    @assert numind(src) == N₁ + N₂
    @assert numout(src) == length(levels[1]) && numin(src) == length(levels[2])
    @assert TupleTools.isperm((p[1]..., p[2]...))
    return _fsbraid((src, p, levels))
end

const _FSBraidKey{I, N₁, N₂} = Tuple{<:FusionTreeBlock{I}, Index2Tuple{N₁, N₂}, Index2Tuple}

@cached function _fsbraid(
        key::_FSBraidKey{I, N₁, N₂}
    )::Tuple{FusionTreeBlock{I, N₁, N₂, fusiontreetype(I, N₁, N₂)}, Matrix{sectorscalartype(I)}} where {I, N₁, N₂}
    src, (p1, p2), (l1, l2) = key

    p = linearizepermutation(p1, p2, numout(src), numin(src))
    levels = (l1..., reverse(l2)...)

    dst, U = repartition(src, numind(src))

    if FusionStyle(I) isa UniqueFusion && BraidingStyle(I) isa SymmetricBraiding
        uncoupled′ = TupleTools._permute(dst.uncoupled[1], p)
        isdual′ = TupleTools._permute(dst.isdual[1], p)

        dst′ = FusionTreeBlock{I}(uncoupled′, isdual′)
        U_tmp = transformation_matrix(dst′, dst) do (f₁, f₂)
            return ((f₁′, f₂) => c for (f₁, c) in braid(f₁, p, levels))
        end
        dst = dst′
        U = U_tmp * U
    else
        for s in permutation2swaps(p)
            inv = levels[s] > levels[s + 1]
            dst, U_tmp = artin_braid(dst, s; inv)
            U = U_tmp * U
        end
    end

    if N₂ == 0
        return dst, U
    else
        dst, U_tmp = repartition(dst, N₁)
        U = U_tmp * U
        return dst, U
    end
end

function CacheStyle(::typeof(_fsbraid), k::_FSBraidKey{I}) where {I}
    if FusionStyle(I) isa UniqueFusion
        return NoCache()
    else
        return GlobalLRUCache()
    end
end

function permute(src::FusionTreeBlock{I}, p::Index2Tuple) where {I}
    @assert BraidingStyle(I) isa SymmetricBraiding
    levels1 = ntuple(identity, numout(src))
    levels2 = numout(src) .+ ntuple(identity, numin(src))
    return braid(src, p, (levels1, levels2))
end
