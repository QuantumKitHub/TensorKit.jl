# Index manipulations
#---------------------

# find the scalartype after applying operations: take into account fusion and/or braiding
# might need to become Float or Complex to capture complex recoupling coefficients but don't alter precision
for (operation, manipulation) in (
        :flip => :sector, :twist => :braiding,
        :transpose => :fusion, :permute => :sector, :braid => :sector,
    )
    promote_op = Symbol(:promote_, operation)
    manipulation_scalartype = Symbol(manipulation, :scalartype)

    @eval begin
        $promote_op(t::AbstractTensorMap) = $promote_op(typeof(t))
        $promote_op(::Type{T}) where {T <: AbstractTensorMap} =
            $promote_op(scalartype(T), sectortype(T))
        $promote_op(::Type{T}, ::Type{I}) where {T <: Number, I <: Sector} =
            sectorscalartype(I) <: Integer ? T :
            sectorscalartype(I) <: Real ? float(T) : complex(T)
        # TODO: currently the manipulations all use sectorscalartype, change to:
        # $manipulation_scalartype(I) <: Integer ? T :
        # $manipulation_scalartype(I) <: Real ? float(T) : complex(T)
    end
end

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
    t′ = similar(t, promote_flip(t), P)
    for (f₁, f₂) in fusiontrees(t)
        (f₁′, f₂′), factor = only(flip((f₁, f₂), I; inv))
        scale!(t′[f₁′, f₂′], t[f₁, f₂], factor)
    end
    return t′
end

# --------------
#   permute(!)
# --------------
"""
    permute!(tdst, tsrc, (p₁, p₂)::Index2Tuple, α = 1, β = 0, [backend], [allocator]) -> tdst

Compute `tdst = β * tdst + α * permute(tsrc, (p₁, p₂))`, writing the result into `tdst`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.
Optionally specify a `backend` and `allocator` for the underlying array operation.

See also [`permute`](@ref) for creating a new tensor.
"""
@propagate_inbounds function Base.permute!(
        tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p::Index2Tuple,
        α::Number = One(), β::Number = Zero(),
        backend::AbstractBackend = TO.DefaultBackend(), allocator = TO.DefaultAllocator()
    )
    @boundscheck spacecheck_transform(permute, tdst, tsrc, p)
    transformer = treepermuter(tdst, tsrc, p)
    return @inbounds add_transform!(tdst, tsrc, p, transformer, α, β, backend, allocator)
end

"""
    permute(tsrc, (p₁, p₂)::Index2Tuple; copy = false, [backend], [allocator]) -> tdst

Return tensor `tdst` obtained by permuting the indices of `tsrc`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.

If `copy = false`, `tdst` might share data with `tsrc` whenever possible.
Otherwise, a copy is always made.
Optionally specify a `backend` and `allocator` for the underlying array operation.

See also [`permute!`](@ref) for writing into an existing destination.
"""
function permute(
        t::AbstractTensorMap, p::Index2Tuple;
        copy::Bool = false, backend::AbstractBackend = TO.DefaultBackend(), allocator = TO.DefaultAllocator()
    )
    # share data if possible
    if !copy
        if p == (codomainind(t), domainind(t))
            return t
        elseif t isa TensorMap && has_shared_permute(t, p)
            return TensorMap(t.data, permute(space(t), p))
        end
    end

    # general case
    tdst = similar(t, promote_permute(t), permute(space(t), p))
    return @inbounds permute!(tdst, t, p, One(), Zero(), backend, allocator)
end
function permute(t::AdjointTensorMap, (p₁, p₂)::Index2Tuple; kwargs...)
    p₁′ = adjointtensorindices(t, p₂)
    p₂′ = adjointtensorindices(t, p₁)
    return adjoint(permute(adjoint(t), (p₁′, p₂′); kwargs...))
end
permute(t::AbstractTensorMap, p::IndexTuple; kwargs...) = permute(t, (p, ()); kwargs...)

function has_shared_permute(t::AbstractTensorMap, (p₁, p₂)::Index2Tuple)
    return (p₁ === codomainind(t) && p₂ === domainind(t))
end
function has_shared_permute(t::TensorMap, (p₁, p₂)::Index2Tuple)
    if p₁ === codomainind(t) && p₂ === domainind(t)
        return true
    elseif sectortype(t) === Trivial
        stridet = Base.Fix1(stride, t[])
        sizet = Base.Fix1(size, t[])
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

# -------------
#   braid(!)
# -------------
"""
    braid!(tdst, tsrc, (p₁, p₂)::Index2Tuple, levels::IndexTuple, α = 1, β = 0, [backend], [allocator]) -> tdst

Compute `tdst = β * tdst + α * braid(tsrc, (p₁, p₂), levels)`, writing the result into `tdst`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.
Here, `levels` is a tuple of length `numind(tsrc)` that assigns a level or height to the indices of `tsrc`,
which determines whether they will braid over or under any other index with which they have to change places.
Optionally specify a `backend` and `allocator` for the underlying array operation.

See also [`braid`](@ref) for creating a new tensor.
"""
@propagate_inbounds function braid!(
        tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p::Index2Tuple, levels::IndexTuple,
        α::Number = One(), β::Number = Zero(),
        backend::AbstractBackend = TO.DefaultBackend(), allocator = TO.DefaultAllocator()
    )
    @boundscheck spacecheck_transform(braid, tdst, tsrc, p, levels)
    levels1 = TupleTools.getindices(levels, codomainind(tsrc))
    levels2 = TupleTools.getindices(levels, domainind(tsrc))
    transformer = treebraider(tdst, tsrc, p, (levels1, levels2))
    return @inbounds add_transform!(tdst, tsrc, p, transformer, α, β, backend, allocator)
end

"""
    braid(tsrc, (p₁, p₂)::Index2Tuple, levels::IndexTuple; copy=false,
          backend=DefaultBackend(), allocator=DefaultAllocator()) -> tdst::TensorMap

Return tensor `tdst` obtained by braiding the indices of `tsrc`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.
Here, `levels` is a tuple of length `numind(tsrc)` that assigns a level or height to the indices of `tsrc`,
which determines whether they will braid over or under any other index with which they have to change places.

If `copy=false`, `tdst` might share data with `tsrc` whenever possible. Otherwise, a copy is always made.
Optionally specify a `backend` and `allocator` for the underlying array operation.

See also [`braid!`](@ref) for writing into an existing destination.
"""
function braid(
        t::AbstractTensorMap, p::Index2Tuple, levels::IndexTuple;
        copy::Bool = false, backend::AbstractBackend = TO.DefaultBackend(), allocator = TO.DefaultAllocator()
    )
    length(levels) == numind(t) || throw(ArgumentError(lazy"length of levels should be $(numind(t)), got $(length(levels))"))

    BraidingStyle(sectortype(t)) isa SymmetricBraiding && return permute(t, p; copy, backend, allocator)
    (!copy && p == (codomainind(t), domainind(t))) && return t

    # general case
    tdst = similar(t, promote_braid(t), permute(space(t), p))
    return @inbounds braid!(tdst, t, p, levels, One(), Zero(), backend, allocator)
end
function braid(
        t::AdjointTensorMap, (p₁, p₂)::Index2Tuple, levels::IndexTuple;
        kwargs...
    )
    p₁′ = adjointtensorindices(t, p₂)
    p₂′ = adjointtensorindices(t, p₁)
    perm = adjointtensorindices(adjoint(t), ntuple(identity, numind(t)))
    levels′ = TupleTools.getindices(levels, perm)
    return adjoint(braid(adjoint(t), (p₁′, p₂′), levels′; kwargs...))
end

# ----------------
#   transpose(!)
# ----------------
_transpose_indices(t::AbstractTensorMap) = (reverse(domainind(t)), reverse(codomainind(t)))

"""
    transpose!(tdst, tsrc, (p₁, p₂)::Index2Tuple, α = 1, β = 0, [backend], [allocator]) -> tdst

Compute `tdst = β * tdst + α * transpose(tsrc, (p₁, p₂))`, writing the result into `tdst`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.
The new index positions should be attainable without any indices crossing each other, i.e.,
the permutation `(p₁..., reverse(p₂)...)` should constitute a cyclic permutation of
`(codomainind(tsrc)..., reverse(domainind(tsrc))...)`.
Optionally specify a `backend` and `allocator` for the underlying array operation.

See also [`transpose`](@ref) for creating a new tensor.
"""
function LinearAlgebra.transpose!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap)
    return transpose!(tdst, tsrc, _transpose_indices(tsrc))
end
@propagate_inbounds function LinearAlgebra.transpose!(
        tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p::Index2Tuple,
        α::Number = One(), β::Number = Zero(),
        backend::AbstractBackend = TO.DefaultBackend(), allocator = TO.DefaultAllocator()
    )
    @boundscheck spacecheck_transform(transpose, tdst, tsrc, p)
    transformer = treetransposer(tdst, tsrc, p)
    return @inbounds add_transform!(tdst, tsrc, p, transformer, α, β, backend, allocator)
end

"""
    transpose(tsrc, (p₁, p₂)::Index2Tuple; copy=false,
              backend=DefaultBackend(), allocator=DefaultAllocator()) -> tdst::TensorMap

Return tensor `tdst` obtained by transposing the indices of `tsrc`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.
The new index positions should be attainable without any indices crossing each other, i.e.,
the permutation `(p₁..., reverse(p₂)...)` should constitute a cyclic permutation of
`(codomainind(tsrc)..., reverse(domainind(tsrc))...)`.

If `copy=false`, `tdst` might share data with `tsrc` whenever possible. Otherwise, a copy is always made.
Optionally specify a `backend` and `allocator` for the underlying array operation.

See also [`transpose!`](@ref) for writing into an existing destination.
"""
function LinearAlgebra.transpose(
        t::AbstractTensorMap, p::Index2Tuple = _transpose_indices(t);
        copy::Bool = false, backend = TO.DefaultBackend(), allocator = TO.DefaultAllocator()
    )
    sectortype(t) === Trivial && return permute(t, p; copy, backend, allocator)
    (!copy && p == (codomainind(t), domainind(t))) && return t

    # general case
    tdst = similar(t, promote_transpose(t), permute(space(t), p))
    return @inbounds transpose!(tdst, t, p, One(), Zero(), backend, allocator)
end

function LinearAlgebra.transpose(
        t::AdjointTensorMap, (p₁, p₂)::Index2Tuple = _transpose_indices(t);
        copy::Bool = false, backend = TO.DefaultBackend(), allocator = TO.DefaultAllocator()
    )
    p₁′ = map(n -> adjointtensorindex(t, n), p₂)
    p₂′ = map(n -> adjointtensorindex(t, n), p₁)
    return adjoint(transpose(adjoint(t), (p₁′, p₂′); copy, backend, allocator))
end

# -------------------
#   repartition(!)
# -------------------
"""
    repartition!(tdst, tsrc, α = 1, β = 0, [backend], [allocator]) -> tdst

Compute `tdst = β * tdst + α * repartition(tsrc)`, writing the result into `tdst`.
This is a special case of `transpose!` that only changes the partition of indices between
codomain and domain, without changing their cyclic order.
Optionally specify a `backend` and `allocator` for the underlying array operation.

See also [`repartition`](@ref) for creating a new tensor.
"""
@propagate_inbounds function repartition!(
        tdst::AbstractTensorMap, tsrc::AbstractTensorMap,
        α::Number = One(), β::Number = Zero(),
        backend::AbstractBackend = TO.DefaultBackend(), allocator = TO.DefaultAllocator()
    )
    check_spacetype(tdst, tsrc)
    numind(tsrc) == numind(tdst) ||
        throw(ArgumentError("tsrc and tdst should have an equal amount of indices"))
    all_inds = (codomainind(tsrc)..., reverse(domainind(tsrc))...)
    p₁ = ntuple(i -> all_inds[i], numout(tdst))
    p₂ = reverse(ntuple(i -> all_inds[i + numout(tdst)], numin(tdst)))
    return transpose!(tdst, tsrc, (p₁, p₂), α, β, backend, allocator)
end

"""
    repartition(tsrc, N₁::Int, N₂::Int=numind(tsrc)-N₁; copy=false,
                backend=DefaultBackend(), allocator=DefaultAllocator()) -> tdst

Return tensor `tdst` obtained by repartitioning the indices of `tsrc`.
The codomain and domain of `tdst` correspond to the first `N₁` and last `N₂` spaces of `tsrc`,
respectively.

If `copy=false`, `tdst` might share data with `tsrc` whenever possible. Otherwise, a copy is always made.
Optionally specify a `backend` and `allocator` for the underlying array operation.

See also [`repartition!`](@ref) for writing into an existing destination.
"""
@constprop :aggressive function repartition(
        t::AbstractTensorMap, N₁::Int, N₂::Int = numind(t) - N₁;
        copy::Bool = false, backend = TO.DefaultBackend(), allocator = TO.DefaultAllocator()
    )
    N₁ + N₂ == numind(t) ||
        throw(ArgumentError("Invalid repartition: $(numind(t)) to ($N₁, $N₂)"))
    all_inds = (codomainind(t)..., reverse(domainind(t))...)
    p₁ = ntuple(i -> all_inds[i], N₁)
    p₂ = reverse(ntuple(i -> all_inds[i + N₁], N₂))
    return transpose(t, (p₁, p₂); copy, backend, allocator)
end

# Twist
function has_shared_twist(t, inds)
    I = sectortype(t)
    if BraidingStyle(I) == NoBraiding()
        for i in inds
            cs = sectors(space(t, i))
            all(isunit, cs) || throw(SectorMismatch(lazy"Cannot twist sectors $cs"))
        end
        return true
    elseif BraidingStyle(I) == Bosonic()
        return true
    else
        for i in inds
            cs = sectors(space(t, i))
            all(isone ∘ twist, cs) || return false
        end
        return true
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
    (scalartype(t) <: Real && !(sectorscalartype(sectortype(t)) <: Real)) &&
        throw(ArgumentError("Can't in-place twist a real tensor with complex sector type"))
    has_shared_twist(t, inds) && return t

    (scalartype(t) <: Real && !(sectorscalartype(sectortype(t)) <: Real)) &&
        throw(ArgumentError("No in-place `twist!` for a real tensor with complex sector type"))

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
    tdst = similar(t, promote_twist(t))
    copy!(tdst, t)
    return twist!(tdst, inds; inv)
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
spacecheck_transform(f, tdst::AbstractTensorMap, tsrc::AbstractTensorMap, args...) =
    spacecheck_transform(f, space(tdst), space(tsrc), args...)
@noinline function spacecheck_transform(f, Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple)
    check_spacetype(Vdst, Vsrc)
    f(Vsrc, p) == Vdst ||
        throw(
        SpaceMismatch(
            lazy"""
            incompatible spaces for `$f(Vsrc, $p) -> Vdst`
            Vsrc = $Vsrc
            Vdst = $Vdst
            """
        )
    )
    return nothing
end
@noinline function spacecheck_transform(::typeof(braid), Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple, levels::IndexTuple)
    check_spacetype(Vdst, Vsrc)
    braid(Vsrc, p, levels) == Vdst ||
        throw(
        SpaceMismatch(
            lazy"""
            incompatible spaces for `braid(Vsrc, $p, $levels) -> Vdst`
            Vsrc = $Vsrc
            Vdst = $Vdst
            """
        )
    )
    return nothing
end

# Deprecated add_*! wrappers
# --------------------------
Base.@deprecate(
    add_permute!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p::Index2Tuple, α::Number, β::Number, backend::AbstractBackend...),
    permute!(tdst, tsrc, p, α, β, backend...)
)
Base.@deprecate(
    add_braid!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p::Index2Tuple, levels::IndexTuple, α::Number, β::Number, backend::AbstractBackend...),
    braid!(tdst, tsrc, p, levels, α, β, backend...)
)
Base.@deprecate(
    add_transpose!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p::Index2Tuple, α::Number, β::Number, backend::AbstractBackend...),
    transpose!(tdst, tsrc, p, α, β, backend...)
)

@propagate_inbounds function add_transform!(
        tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p::Index2Tuple, transformer,
        α::Number, β::Number, backend, allocator
    )
    # `permute` is used as a stand-in for all index rearrangements here: permute, braid, and
    # transpose all produce the same destination space for a given permutation tuple `p`.
    @boundscheck spacecheck_transform(permute, tdst, tsrc, p)

    # Three cases, from cheapest to most expensive:
    #   1. trivial permutation: delegate to `add!` which handles α/β scaling directly
    #   2. Trivial sector type: no fusion tree bookkeeping, call tensoradd! on the raw array
    #   3. general case: iterate over (blocks of) fusion trees, potentially multi-threaded
    if p[1] === codomainind(tsrc) && p[2] === domainind(tsrc)
        add!(tdst, tsrc, α, β)
    else
        I = sectortype(tdst)
        if I === Trivial
            TO.tensoradd!(tdst[], tsrc[], p, false, α, β, backend, allocator)
        else
            ntasks = use_threaded_transform(tdst, transformer) ? get_num_transformer_threads() : 1
            scheduler = ntasks == 1 ? SerialScheduler() : DynamicScheduler(; ntasks, split = :roundrobin)
            if tdst isa TensorMap && tsrc isa TensorMap # unpack data fields to avoid specializing
                add_transform_kernel!(tdst.data, tsrc.data, p, transformer, α, β, backend, allocator, scheduler)
            else
                add_transform_kernel!(tdst, tsrc, p, transformer, α, β, backend, allocator, scheduler)
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

function add_transform_kernel!(
        tdst, tsrc, p, transformer, α, β, backend, allocator, scheduler
    )
    I = sectortype(tdst)
    if FusionStyle(I) === UniqueFusion()
        # Abelian / unique-fusion: each source fusion tree pair (f₁, f₂) maps to exactly
        # one destination pair (f₁′, f₂′) with a scalar coefficient. No mixing occurs.
        tforeach(fusiontrees(tsrc); scheduler) do (f₁, f₂)
            (f₁′, f₂′), coeff = transformer((f₁, f₂))
            @inbounds TO.tensoradd!(
                tdst[f₁′, f₂′], tsrc[f₁, f₂],
                p, false, α * coeff, β, backend, allocator
            )
        end
    else
        # Non-Abelian fusion: trees sharing the same set of uncoupled (external) sectors
        # form a *fusion block* and mix under the transformation via a recoupling matrix U
        # (rows = destination trees, columns = source trees). We iterate over blocks.
        tl_buffers = OhMyThreads.TaskLocalValue(() -> allocate_buffers(tdst, tsrc, transformer, allocator))
        tforeach(fusionblocks(tsrc); scheduler) do src
            dst, U = transformer(src)
            if length(src) == 1
                # Degenerate block: single tree, U is a 1×1 scalar — skip the buffer + matmul.
                (f₁, f₂) = only(fusiontrees(src))
                (f₁′, f₂′) = only(fusiontrees(dst))
                @inbounds TO.tensoradd!(
                    tdst[f₁′, f₂′], tsrc[f₁, f₂],
                    p, false, α * only(U), β, backend, allocator
                )
            else
                # Multi-tree block: apply recoupling via a three-step pack → matmul → unpack.
                #   1. Extract: flatten each source block into a column of buffer_src
                #      (shape blocksize × cols), using a trivial permutation so that the
                #      index layout is canonical before the matmul.
                #   2. Recoupling: buffer_dst = buffer_src * U^T  (blocksize × rows)
                #   3. Insert: scatter columns of buffer_dst to destination blocks,
                #      applying the actual permutation p in the same step.
                buffer1, buffer2 = tl_buffers[]
                rows, cols = size(U)
                sz_src = size(tsrc[first(fusiontrees(src))...])
                blocksize = prod(sz_src)
                ptriv = (ntuple(identity, length(sz_src)), ())
                buffer_src = StridedView(buffer2, (blocksize, cols), (1, blocksize), 0)
                @inbounds for (i, (f₁, f₂)) in enumerate(fusiontrees(src))
                    TO.tensoradd!(
                        sreshape(buffer_src[:, i], sz_src), tsrc[f₁, f₂],
                        ptriv, false, One(), Zero(), backend, allocator
                    )
                end
                buffer_dst = StridedView(buffer1, (blocksize, rows), (1, blocksize), 0)
                mul!(buffer_dst, buffer_src, transpose(StridedView(U)))
                @inbounds for (i, (f₃, f₄)) in enumerate(fusiontrees(dst))
                    TO.tensoradd!(
                        tdst[f₃, f₄], sreshape(buffer_dst[:, i], sz_src),
                        p, false, α, β, backend, allocator
                    )
                end
            end
        end
    end
    return nothing
end

# TensorMap specializations: operate directly on the flat data vector to avoid
# repeated dictionary lookups into t.data. The transformer has precomputed all
# StridedView descriptors (size, offset, strides) for each fusion tree block.
# No symmetry types left -- no repeated specialization needed
function add_transform_kernel!(
        data_dst::DenseVector, data_src::DenseVector, p, transformer::AbelianTreeTransformer,
        α, β, backend, allocator, scheduler
    )
    # Each entry is (coeff, struct_dst, struct_src) where struct_{dst,src} = (size, offset, strides)
    # locating the block for one fusion tree pair inside the flat data vector.
    tforeach(transformer.data; scheduler) do (coeff, struct_dst, struct_src)
        TO.tensoradd!(
            StridedView(data_dst, struct_dst...), StridedView(data_src, struct_src...),
            p, false, α * coeff, β, backend, allocator
        )
    end
    return nothing
end
function add_transform_kernel!(
        data_dst::DenseVector, data_src::DenseVector, p, transformer::GenericTreeTransformer,
        α, β, backend, allocator, scheduler
    )
    # Each entry covers one fusion block:
    #   U            — recoupling matrix (rows = dst trees, cols = src trees)
    #   sz_{dst,src} — array shape of each block (same for all trees in the block)
    #   structs_{dst,src}[i] — (offset, strides) into the flat data vector for tree i
    tl_buffers = OhMyThreads.TaskLocalValue(() -> allocate_buffers(data_dst, data_src, transformer, allocator))
    tforeach(transformer.data; scheduler) do (U, (sz_dst, structs_dst), (sz_src, structs_src))
        if length(U) == 1
            # Degenerate block with a single tree: no matmul needed.
            coeff = only(U)
            TO.tensoradd!(
                StridedView(data_dst, sz_dst, only(structs_dst)...),
                StridedView(data_src, sz_src, only(structs_src)...),
                p, false, α * coeff, β, backend, allocator
            )
        else
            # Multi-tree block: pack → recoupling matmul → unpack.
            # buffer2 = source staging area, buffer1 = destination staging area.
            buffer1, buffer2 = tl_buffers[]
            rows, cols = size(U)
            blocksize = prod(sz_src)
            ptriv = (ntuple(identity, length(sz_src)), ())

            # 1. Extract: copy each source block into column i of buffer_src as a flat vector,
            #    using a trivial permutation so the layout is canonical before the matmul.
            buffer_src = StridedView(buffer2, (blocksize, cols), (1, blocksize), 0)
            @inbounds for (i, struct_src_i) in enumerate(structs_src)
                TO.tensoradd!(
                    sreshape(buffer_src[:, i], sz_src), StridedView(data_src, sz_src, struct_src_i...),
                    ptriv, false, One(), Zero(), backend, allocator
                )
            end

            # 2. Recoupling: buffer_dst = buffer_src * U^T  (each output tree is a linear
            #    combination of input trees weighted by the recoupling coefficients).
            buffer_dst = StridedView(buffer1, (blocksize, rows), (1, blocksize), 0)
            mul!(buffer_dst, buffer_src, transpose(StridedView(U)))

            # 3. Insert: scatter column i of buffer_dst into the destination, applying the
            #    actual index permutation p in the same tensoradd! call.
            @inbounds for (i, struct_dst_i) in enumerate(structs_dst)
                TO.tensoradd!(
                    StridedView(data_dst, sz_dst, struct_dst_i...), sreshape(buffer_dst[:, i], sz_src),
                    p, false, α, β, backend, allocator
                )
            end
        end
    end
    return nothing
end
