# Index manipulations
#---------------------
"""
    flip(t::AbstractTensorMap, I) -> t′::AbstractTensorMap

Return a new tensor that is isomorphic to `t` but where the arrows on the indices `i` that satisfy
`i ∈ I` are flipped, i.e. `space(t′, i) = flip(space(t, i))`.

!!! note
    The isomorphism that `flip` applies to each of the indices `i ∈ I` is such that flipping two indices
    that are afterwards contracted within an `@tensor` contraction will yield the same result as without
    flipping those indices first. However, `flip` is not involutory, i.e. `flip(flip(t, I), I) != t` in
    general. To obtain the original tensor, one can use the `inv` keyword, i.e. it holds that
    `flip(flip(t, I), I; inv=true) == t`.
"""
function flip(t::AbstractTensorMap, I; inv::Bool = false)
    P = flip(space(t), I)
    t′ = similar(t, P)
    for (f₁, f₂) in fusiontrees(t)
        (f₁′, f₂′), factor = only(flip(f₁, f₂, I; inv))
        scale!(t′[f₁′, f₂′], t[f₁, f₂], factor)
    end
    return t′
end

"""
    permute!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, (p₁, p₂)::Index2Tuple)
        -> tdst

Write into `tdst` the result of permuting the indices of `tsrc`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.
                
See [`permute`](@ref) for creating a new tensor and [`add_permute!`](@ref) for a more general version.
"""
@propagate_inbounds function Base.permute!(
        tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p::Index2Tuple
    )
    return add_permute!(tdst, tsrc, p, One(), Zero())
end

"""
    permute(tsrc::AbstractTensorMap, (p₁, p₂)::Index2Tuple;
            copy::Bool=false)
        -> tdst::TensorMap

Return tensor `tdst` obtained by permuting the indices of `tsrc`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.

If `copy=false`, `tdst` might share data with `tsrc` whenever possible. Otherwise, a copy is always made.

To permute into an existing destination, see [permute!](@ref) and [`add_permute!`](@ref)
"""
function permute(
        t::AbstractTensorMap, (p₁, p₂)::Index2Tuple{N₁, N₂}; copy::Bool = false
    ) where {N₁, N₂}
    space′ = permute(space(t), (p₁, p₂))
    # share data if possible
    if !copy && p₁ === codomainind(t) && p₂ === domainind(t)
        return t
    end
    # general case
    @inbounds begin
        return permute!(similar(t, space′), t, (p₁, p₂))
    end
end
function permute(t::TensorMap, (p₁, p₂)::Index2Tuple{N₁, N₂}; copy::Bool = false) where {N₁, N₂}
    space′ = permute(space(t), (p₁, p₂))
    # share data if possible
    if !copy
        if p₁ === codomainind(t) && p₂ === domainind(t)
            return t
        elseif has_shared_permute(t, (p₁, p₂))
            return TensorMap(t.data, space′)
        end
    end
    # general case
    @inbounds begin
        return permute!(similar(t, space′), t, (p₁, p₂))
    end
end
function permute(t::AdjointTensorMap, (p₁, p₂)::Index2Tuple; copy::Bool = false)
    p₁′ = adjointtensorindices(t, p₂)
    p₂′ = adjointtensorindices(t, p₁)
    return adjoint(permute(adjoint(t), (p₁′, p₂′); copy))
end
function permute(t::AbstractTensorMap, p::IndexTuple; copy::Bool = false)
    return permute(t, (p, ()); copy)
end

function has_shared_permute(t::AbstractTensorMap, (p₁, p₂)::Index2Tuple)
    return (p₁ === codomainind(t) && p₂ === domainind(t))
end
function has_shared_permute(t::TensorMap, (p₁, p₂)::Index2Tuple)
    if p₁ === codomainind(t) && p₂ === domainind(t)
        return true
    elseif sectortype(t) === Trivial
        stridet = i -> stride(t[], i)
        sizet = i -> size(t[], i)
        canfuse1, d1, s1 = TO._canfuse(sizet.(p₁), stridet.(p₁))
        canfuse2, d2, s2 = TO._canfuse(sizet.(p₂), stridet.(p₂))
        return canfuse1 && canfuse2 && s1 == 1 && (d2 == 1 || s2 == d1)
    else
        return false
    end
end
function has_shared_permute(t::AdjointTensorMap, (p₁, p₂)::Index2Tuple)
    p₁′ = adjointtensorindices(t, p₂)
    p₂′ = adjointtensorindices(t, p₁)
    return has_shared_permute(t', (p₁′, p₂′))
end

# Braid
"""
    braid!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap,
           (p₁, p₂)::Index2Tuple, levels::Tuple)
        -> tdst

Write into `tdst` the result of braiding the indices of `tsrc`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.
Here, `levels` is a tuple of length `numind(tsrc)` that assigns a level or height to the indices of `tsrc`,
which determines whether they will braid over or under any other index with which they have to change places.

See [`braid`](@ref) for creating a new tensor and [`add_braid!`](@ref) for a more general version.
"""
@propagate_inbounds function braid!(
        tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p::Index2Tuple, levels::IndexTuple
    )
    return add_braid!(tdst, tsrc, p, levels, One(), Zero())
end

"""
    braid(tsrc::AbstractTensorMap, (p₁, p₂)::Index2Tuple, levels::IndexTuple;
          copy::Bool = false)
        -> tdst::TensorMap

Return tensor `tdst` obtained by braiding the indices of `tsrc`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.
Here, `levels` is a tuple of length `numind(tsrc)` that assigns a level or height to the indices of `tsrc`,
which determines whether they will braid over or under any other index with which they have to change places.

If `copy=false`, `tdst` might share data with `tsrc` whenever possible. Otherwise, a copy is always made.

To braid into an existing destination, see [braid!](@ref) and [`add_braid!`](@ref)
"""
function braid(
        t::AbstractTensorMap, (p₁, p₂)::Index2Tuple, levels::IndexTuple; copy::Bool = false
    )
    @assert length(levels) == numind(t)
    if BraidingStyle(sectortype(t)) isa SymmetricBraiding
        return permute(t, (p₁, p₂); copy = copy)
    end
    if !copy && p₁ == codomainind(t) && p₂ == domainind(t)
        return t
    end
    # general case
    space′ = permute(space(t), (p₁, p₂))
    @inbounds begin
        return braid!(similar(t, space′), t, (p₁, p₂), levels)
    end
end
# TODO: braid for `AdjointTensorMap`; think about how to map the `levels` argument.

# Transpose
_transpose_indices(t::AbstractTensorMap) = (reverse(domainind(t)), reverse(codomainind(t)))

"""
    transpose!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap,
               (p₁, p₂)::Index2Tuple)
        -> tdst

Write into `tdst` the result of transposing the indices of `tsrc`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.
The new index positions should be attainable without any indices crossing each other, i.e.,
the permutation `(p₁..., reverse(p₂)...)` should constitute a cyclic permutation of `(codomainind(tsrc)..., reverse(domainind(tsrc))...)`.

See [`transpose`](@ref) for creating a new tensor and [`add_transpose!`](@ref) for a more general version.
"""
function LinearAlgebra.transpose!(
        tdst::AbstractTensorMap, tsrc::AbstractTensorMap, (p₁, p₂)::Index2Tuple = _transpose_indices(t)
    )
    return add_transpose!(tdst, tsrc, (p₁, p₂), One(), Zero())
end

"""
    transpose(tsrc::AbstractTensorMap, (p₁, p₂)::Index2Tuple;
              copy::Bool=false)
        -> tdst::TensorMap

Return tensor `tdst` obtained by transposing the indices of `tsrc`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.
The new index positions should be attainable without any indices crossing each other, i.e.,
the permutation `(p₁..., reverse(p₂)...)` should constitute a cyclic permutation of `(codomainind(tsrc)..., reverse(domainind(tsrc))...)`.

If `copy=false`, `tdst` might share data with `tsrc` whenever possible. Otherwise, a copy is always made.

To permute into an existing destination, see [permute!](@ref) and [`add_permute!`](@ref)
"""
function LinearAlgebra.transpose(
        t::AbstractTensorMap, (p₁, p₂)::Index2Tuple = _transpose_indices(t);
        copy::Bool = false
    )
    if sectortype(t) === Trivial
        return permute(t, (p₁, p₂); copy = copy)
    end
    if !copy && p₁ == codomainind(t) && p₂ == domainind(t)
        return t
    end
    # general case
    space′ = permute(space(t), (p₁, p₂))
    @inbounds begin
        return transpose!(similar(t, space′), t, (p₁, p₂))
    end
end

function LinearAlgebra.transpose(
        t::AdjointTensorMap, (p₁, p₂)::Index2Tuple = _transpose_indices(t);
        copy::Bool = false
    )
    p₁′ = map(n -> adjointtensorindex(t, n), p₂)
    p₂′ = map(n -> adjointtensorindex(t, n), p₁)
    return adjoint(transpose(adjoint(t), (p₁′, p₂′); copy = copy))
end

"""
    repartition!(tdst::AbstractTensorMap{S}, tsrc::AbstractTensorMap{S}) where {S} -> tdst

Write into `tdst` the result of repartitioning the indices of `tsrc`. This is just a special
case of a transposition that only changes the number of in- and outgoing indices.

See [`repartition`](@ref) for creating a new tensor.
"""
function repartition!(tdst::AbstractTensorMap{S}, tsrc::AbstractTensorMap{S}) where {S}
    numind(tsrc) == numind(tdst) ||
        throw(ArgumentError("tsrc and tdst should have an equal amount of indices"))
    all_inds = (codomainind(tsrc)..., reverse(domainind(tsrc))...)
    p₁ = ntuple(i -> all_inds[i], numout(tdst))
    p₂ = reverse(ntuple(i -> all_inds[i + numout(tdst)], numin(tdst)))
    return transpose!(tdst, tsrc, (p₁, p₂))
end

"""
    repartition(tsrc::AbstractTensorMap{S}, N₁::Int, N₂::Int; copy::Bool=false) where {S}
        -> tdst::AbstractTensorMap{S,N₁,N₂}

Return tensor `tdst` obtained by repartitioning the indices of `t`.
The codomain and domain of `tdst` correspond to the first `N₁` and last `N₂` spaces of `t`,
respectively.

If `copy=false`, `tdst` might share data with `tsrc` whenever possible. Otherwise, a copy is always made.

To repartition into an existing destination, see [repartition!](@ref).
"""
@constprop :aggressive function repartition(
        t::AbstractTensorMap, N₁::Int, N₂::Int = numind(t) - N₁; copy::Bool = false
    )
    N₁ + N₂ == numind(t) ||
        throw(ArgumentError("Invalid repartition: $(numind(t)) to ($N₁, $N₂)"))
    all_inds = (codomainind(t)..., reverse(domainind(t))...)
    p₁ = ntuple(i -> all_inds[i], N₁)
    p₂ = reverse(ntuple(i -> all_inds[i + N₁], N₂))
    return transpose(t, (p₁, p₂); copy)
end

# Twist
function has_shared_twist(t, inds)
    I = sectortype(t)
    if BraidingStyle(I) == NoBraiding()
        for i in inds
            cs = sectors(space(t, i))
            all(isone, cs) || throw(SectorMismatch(lazy"Cannot twist sectors $cs"))
        end
        return true
    elseif BraidingStyle(I) == Bosonic()
        return true
    else
        return isempty(inds)
    end
end

"""
    twist!(t::AbstractTensorMap, i::Int; inv::Bool=false) -> t
    twist!(t::AbstractTensorMap, inds; inv::Bool=false) -> t

Apply a twist to the `i`th index of `t`, or all indices in `inds`, storing the result in `t`.
If `inv=true`, use the inverse twist.

See [`twist`](@ref) for creating a new tensor.
"""
function twist!(t::AbstractTensorMap, inds; inv::Bool = false)
    if !all(in(allind(t)), inds)
        msg = "Can't twist indices $inds of a tensor with only $(numind(t)) indices."
        throw(ArgumentError(msg))
    end
    has_shared_twist(t, inds) && return t
    N₁ = numout(t)
    for (f₁, f₂) in fusiontrees(t)
        θ = prod(i -> i <= N₁ ? twist(f₁.uncoupled[i]) : twist(f₂.uncoupled[i - N₁]), inds)
        inv && (θ = θ')
        scale!(t[f₁, f₂], θ)
    end
    return t
end

"""
    twist(tsrc::AbstractTensorMap, i::Int; inv::Bool = false, copy::Bool = false) -> tdst
    twist(tsrc::AbstractTensorMap, inds; inv::Bool = false, copy::Bool = false) -> tdst

Apply a twist to the `i`th index of `tsrc` and return the result as a new tensor.
If `inv = true`, use the inverse twist.
If `copy = false`, `tdst` might share data with `tsrc` whenever possible. Otherwise, a copy is always made.

See [`twist!`](@ref) for storing the result in place.
"""
function twist(t::AbstractTensorMap, inds; inv::Bool = false, copy::Bool = false)
    !copy && has_shared_twist(t, inds) && return t
    return twist!(copy(t), is; inv)
end

# Methods which change the number of indices, implement using `Val(i)` for type inference
"""
    insertleftunit(tsrc::AbstractTensorMap, i=numind(t) + 1;
                   conj=false, dual=false, copy=false) -> tdst

Insert a trivial vector space, isomorphic to the underlying field, at position `i`,
which can be specified as an `Int` or as `Val(i)` for improved type stability.
More specifically, adds a left monoidal unit or its dual.

If `copy=false`, `tdst` might share data with `tsrc` whenever possible. Otherwise, a copy is always made.

See also [`insertrightunit`](@ref insertrightunit(::AbstractTensorMap, ::Val{i}) where {i}),
[`removeunit`](@ref removeunit(::AbstractTensorMap, ::Val{i}) where {i}).
"""
function insertleftunit(
        t::AbstractTensorMap, ::Val{i} = Val(numind(t) + 1);
        copy::Bool = false, conj::Bool = false, dual::Bool = false
    ) where {i}
    W = insertleftunit(space(t), Val(i); conj, dual)
    if t isa TensorMap
        return TensorMap{scalartype(t)}(copy ? Base.copy(t.data) : t.data, W)
    else
        tdst = similar(t, W)
        for (c, b) in blocks(t)
            copy!(block(tdst, c), b)
        end
        return tdst
    end
end

"""
    insertrightunit(tsrc::AbstractTensorMap, i=numind(t);
                    conj=false, dual=false, copy=false) -> tdst

Insert a trivial vector space, isomorphic to the underlying field, after position `i`,
which can be specified as an `Int` or as `Val(i)` for improved type stability.
More specifically, adds a right monoidal unit or its dual.

If `copy=false`, `tdst` might share data with `tsrc` whenever possible. Otherwise, a copy is always made.

See also [`insertleftunit`](@ref insertleftunit(::AbstractTensorMap, ::Val{i}) where {i}),
[`removeunit`](@ref removeunit(::AbstractTensorMap, ::Val{i}) where {i}).
"""
function insertrightunit(
        t::AbstractTensorMap, ::Val{i} = Val(numind(t));
        copy::Bool = false, conj::Bool = false, dual::Bool = false
    ) where {i}
    W = insertrightunit(space(t), Val(i); conj, dual)
    if t isa TensorMap
        return TensorMap{scalartype(t)}(copy ? Base.copy(t.data) : t.data, W)
    else
        tdst = similar(t, W)
        for (c, b) in blocks(t)
            copy!(block(tdst, c), b)
        end
        return tdst
    end
end

"""
    removeunit(tsrc::AbstractTensorMap, i; copy=false) -> tdst

This removes a trivial tensor product factor at position `1 ≤ i ≤ N`, where `i`
can be specified as an `Int` or as `Val(i)` for improved type stability.
For this to work, that factor has to be isomorphic to the field of scalars.

If `copy=false`, `tdst` might share data with `tsrc` whenever possible. Otherwise, a copy is always made.

This operation undoes the work of [`insertleftunit`](@ref insertleftunit(::AbstractTensorMap, ::Val{i}) where {i}) 
and [`insertrightunit`](@ref insertrightunit(::AbstractTensorMap, ::Val{i}) where {i}).
"""
function removeunit(t::AbstractTensorMap, ::Val{i}; copy::Bool = false) where {i}
    W = removeunit(space(t), Val(i))
    if t isa TensorMap
        return TensorMap{scalartype(t)}(copy ? Base.copy(t.data) : t.data, W)
    else
        tdst = similar(t, W)
        for (c, b) in blocks(t)
            copy!(block(tdst, c), b)
        end
        return tdst
    end
end

# Fusing and splitting
# TODO: add functionality for easy fusing and splitting of tensor indices

#-------------------------------------
# Full implementations based on `add`
#-------------------------------------
"""
    add_permute!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, (p₁, p₂)::Index2Tuple,
                 α::Number, β::Number, backend::AbstractBackend...)

Return the updated `tdst`, which is the result of adding `α * tsrc` to `tdst` after permuting 
the indices of `tsrc` according to `(p₁, p₂)`.

See also [`permute`](@ref), [`permute!`](@ref), [`add_braid!`](@ref), [`add_transpose!`](@ref).
"""
@propagate_inbounds function add_permute!(
        tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p::Index2Tuple,
        α::Number, β::Number, backend::AbstractBackend...
    )
    transformer = treepermuter(tdst, tsrc, p)
    return add_transform!(tdst, tsrc, p, transformer, α, β, backend...)
end

"""
    add_braid!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, (p₁, p₂)::Index2Tuple,
               levels::IndexTuple, α::Number, β::Number, backend::AbstractBackend...)

Return the updated `tdst`, which is the result of adding `α * tsrc` to `tdst` after braiding
the indices of `tsrc` according to `(p₁, p₂)` and `levels`.

See also [`braid`](@ref), [`braid!`](@ref), [`add_permute!`](@ref), [`add_transpose!`](@ref).
"""
@propagate_inbounds function add_braid!(
        tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p::Index2Tuple, levels::IndexTuple,
        α::Number, β::Number, backend::AbstractBackend...
    )
    length(levels) == numind(tsrc) ||
        throw(ArgumentError("incorrect levels $levels for tensor map $(codomain(tsrc)) ← $(domain(tsrc))"))

    levels1 = TupleTools.getindices(levels, codomainind(tsrc))
    levels2 = TupleTools.getindices(levels, domainind(tsrc))
    # TODO: arg order for tensormaps is different than for fusiontrees
    transformer = treebraider(tdst, tsrc, p, (levels1, levels2))
    return add_transform!(tdst, tsrc, p, transformer, α, β, backend...)
end

"""
    add_transpose!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, (p₁, p₂)::Index2Tuple,
                   α::Number, β::Number, backend::AbstractBackend...)

Return the updated `tdst`, which is the result of adding `α * tsrc` to `tdst` after transposing
the indices of `tsrc` according to `(p₁, p₂)`.

See also [`transpose`](@ref), [`transpose!`](@ref), [`add_permute!`](@ref), [`add_braid!`](@ref).
"""
@propagate_inbounds function add_transpose!(
        tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p::Index2Tuple,
        α::Number, β::Number, backend::AbstractBackend...
    )
    transformer = treetransposer(tdst, tsrc, p)
    return add_transform!(tdst, tsrc, p, transformer, α, β, backend...)
end

function add_transform!(
        tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p::Index2Tuple, transformer,
        α::Number, β::Number, backend::AbstractBackend...
    )
    @boundscheck begin
        permute(space(tsrc), p) == space(tdst) ||
            throw(SpaceMismatch("source = $(codomain(tsrc))←$(domain(tsrc)),
            dest = $(codomain(tdst))←$(domain(tdst)), p₁ = $(p[1]), p₂ = $(p[2])"))
    end

    if p[1] === codomainind(tsrc) && p[2] === domainind(tsrc)
        add!(tdst, tsrc, α, β)
    else
        I = sectortype(tdst)
        if I === Trivial
            _add_trivial_kernel!(tdst, tsrc, p, transformer, α, β, backend...)
        elseif FusionStyle(I) === UniqueFusion()
            if use_threaded_transform(tdst, transformer)
                _add_abelian_kernel_threaded!(tdst, tsrc, p, transformer, α, β, backend...)
            else
                _add_abelian_kernel_nonthreaded!(
                    tdst, tsrc, p, transformer, α, β, backend...
                )
            end
        else
            if use_threaded_transform(tdst, transformer)
                _add_general_kernel_threaded!(tdst, tsrc, p, transformer, α, β, backend...)
            else
                _add_general_kernel_nonthreaded!(
                    tdst, tsrc, p, transformer, α, β, backend...
                )
            end
        end
    end

    return tdst
end

function use_threaded_transform(t::TensorMap, transformer)
    return get_num_transformer_threads() > 1 && length(t.data) > Strided.MINTHREADLENGTH
end
function use_threaded_transform(t::AbstractTensorMap, transformer)
    return get_num_transformer_threads() > 1 && dim(space(t)) > Strided.MINTHREADLENGTH
end

# Trivial implementations
# -----------------------
function _add_trivial_kernel!(tdst, tsrc, p, transformer, α, β, backend...)
    TO.tensoradd!(tdst[], tsrc[], p, false, α, β, backend...)
    return nothing
end

# Abelian implementations
# -----------------------
function _add_abelian_kernel_nonthreaded!(
        tdst, tsrc, p, transformer::AbelianTreeTransformer, α, β, backend...
    )
    for subtransformer in transformer.data
        _add_transform_single!(tdst, tsrc, p, subtransformer, α, β, backend...)
    end
    return nothing
end

function _add_abelian_kernel_threaded!(
        tdst, tsrc, p, transformer::AbelianTreeTransformer,
        α, β, backend...;
        ntasks::Int = get_num_transformer_threads()
    )
    nblocks = length(transformer.data)
    counter = Threads.Atomic{Int}(1)
    Threads.@sync for _ in 1:min(ntasks, nblocks)
        Threads.@spawn begin
            while true
                local_counter = Threads.atomic_add!(counter, 1)
                local_counter > nblocks && break
                @inbounds subtransformer = transformer.data[local_counter]
                _add_transform_single!(tdst, tsrc, p, subtransformer, α, β, backend...)
            end
        end
    end
    return nothing
end

function _add_transform_single!(
        tdst, tsrc, p, (coeff, struct_dst, struct_src)::_AbelianTransformerData,
        α, β, backend...
    )
    subblock_dst = StridedView(tdst.data, struct_dst...)
    subblock_src = StridedView(tsrc.data, struct_src...)
    TO.tensoradd!(subblock_dst, subblock_src, p, false, α * coeff, β, backend...)
    return nothing
end

function _add_abelian_kernel_nonthreaded!(tdst, tsrc, p, transformer, α, β, backend...)
    for (f₁, f₂) in fusiontrees(tsrc)
        _add_abelian_block!(tdst, tsrc, p, transformer, f₁, f₂, α, β, backend...)
    end
    return nothing
end

function _add_abelian_kernel_threaded!(tdst, tsrc, p, transformer, α, β, backend...)
    Threads.@sync for (f₁, f₂) in fusiontrees(tsrc)
        Threads.@spawn _add_abelian_block!(tdst, tsrc, p, transformer, f₁, f₂, α, β, backend...)
    end
    return nothing
end

function _add_abelian_block!(tdst, tsrc, p, transformer, f₁, f₂, α, β, backend...)
    (f₁′, f₂′), coeff = first(transformer(f₁, f₂))
    @inbounds TO.tensoradd!(tdst[f₁′, f₂′], tsrc[f₁, f₂], p, false, α * coeff, β, backend...)
    return nothing
end

# Non-abelian implementations
# ---------------------------
function _add_general_kernel_nonthreaded!(
        tdst, tsrc, p, transformer::GenericTreeTransformer, α, β, backend...
    )
    # preallocate buffers
    buffers = allocate_buffers(tdst, tsrc, transformer)

    for subtransformer in transformer.data
        # Special case without intermediate buffers whenever there is only a single block
        if length(subtransformer[1]) == 1
            _add_transform_single!(tdst, tsrc, p, subtransformer, α, β, backend...)
        else
            _add_transform_multi!(tdst, tsrc, p, subtransformer, buffers, α, β, backend...)
        end
    end
    return nothing
end

function _add_general_kernel_threaded!(
        tdst, tsrc, p, transformer::GenericTreeTransformer, α, β, backend...;
        ntasks::Int = get_num_transformer_threads()
    )
    nblocks = length(transformer.data)

    counter = Threads.Atomic{Int}(1)
    Threads.@sync for _ in 1:min(ntasks, nblocks)
        Threads.@spawn begin
            # preallocate buffers for each task
            buffers = allocate_buffers(tdst, tsrc, transformer)

            while true
                local_counter = Threads.atomic_add!(counter, 1)
                local_counter > nblocks && break
                @inbounds subtransformer = transformer.data[local_counter]
                if length(subtransformer[1]) == 1
                    _add_transform_single!(tdst, tsrc, p, subtransformer, α, β, backend...)
                else
                    _add_transform_multi!(tdst, tsrc, p, subtransformer, buffers, α, β, backend...)
                end
            end
        end
    end

    return nothing
end

function _add_general_kernel_nonthreaded!(tdst, tsrc, p, transformer, α, β, backend...)
    if iszero(β)
        tdst = zerovector!(tdst)
    elseif !isone(β)
        tdst = scale!(tdst, β)
    end
    for (f₁, f₂) in fusiontrees(tsrc)
        for ((f₁′, f₂′), coeff) in transformer(f₁, f₂)
            @inbounds TO.tensoradd!(tdst[f₁′, f₂′], tsrc[f₁, f₂], p, false, α * coeff, One(), backend...)
        end
    end
    return nothing
end

function _add_transform_single!(
        tdst, tsrc, p, (basistransform, structs_dst, structs_src)::_GenericTransformerData,
        α, β, backend...
    )
    struct_dst = (structs_dst[1], only(structs_dst[2])...)
    struct_src = (structs_src[1], only(structs_src[2])...)
    coeff = only(basistransform)
    _add_transform_single!(tdst, tsrc, p, (coeff, struct_dst, struct_src), α, β, backend...)
    return nothing
end

function _add_transform_multi!(
        tdst, tsrc, p, (basistransform, (sz_dst, structs_dst), (sz_src, structs_src)),
        (buffer1, buffer2), α, β, backend...
    )
    rows, cols = size(basistransform)
    blocksize = prod(sz_src)
    matsize = (
        prod(TupleTools.getindices(sz_src, codomainind(tsrc))),
        prod(TupleTools.getindices(sz_src, domainind(tsrc))),
    )

    # Filling up a buffer with contiguous data
    buffer_src = StridedView(buffer2, (blocksize, cols), (1, blocksize), 0)
    for (i, struct_src) in enumerate(structs_src)
        subblock_src = sreshape(StridedView(tsrc.data, sz_src, struct_src...), matsize)
        _copyto!(buffer_src[:, i], subblock_src)
    end

    # Resummation into a second buffer using BLAS
    buffer_dst = StridedView(buffer1, (blocksize, rows), (1, blocksize), 0)
    mul!(buffer_dst, buffer_src, basistransform, α, Zero())

    # Filling up the output
    for (i, struct_dst) in enumerate(structs_dst)
        subblock_dst = StridedView(tdst.data, sz_dst, struct_dst...)
        bufblock_dst = sreshape(buffer_dst[:, i], sz_src)
        TO.tensoradd!(subblock_dst, bufblock_dst, p, false, One(), β, backend...)
    end

    return nothing
end

function _add_general_kernel_threaded!(tdst, tsrc, p, transformer, α, β, backend...)
    if iszero(β)
        tdst = zerovector!(tdst)
    elseif !isone(β)
        tdst = scale!(tdst, β)
    end
    Threads.@sync for s₁ in sectors(codomain(tsrc)), s₂ in sectors(domain(tsrc))
        Threads.@spawn _add_nonabelian_sector!(tdst, tsrc, p, transformer, s₁, s₂, α, backend...)
    end
    return nothing
end

function _add_nonabelian_sector!(tdst, tsrc, p, fusiontreetransform, s₁, s₂, α, backend...)
    for (f₁, f₂) in fusiontrees(tsrc)
        (f₁.uncoupled == s₁ && f₂.uncoupled == s₂) || continue
        for ((f₁′, f₂′), coeff) in fusiontreetransform(f₁, f₂)
            @inbounds TO.tensoradd!(tdst[f₁′, f₂′], tsrc[f₁, f₂], p, false, α * coeff, One(), backend...)
        end
    end
    return nothing
end
