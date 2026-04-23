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
    length(levels) == numind(t) || throw(ArgumentError("invalid levels"))

    BraidingStyle(sectortype(t)) isa SymmetricBraiding && return permute(t, p; copy, backend, allocator)
    (!copy && p == (codomainind(t), domainind(t))) && return t

    # general case
    tdst = similar(t, promote_braid(t), permute(space(t), p))
    return @inbounds braid!(tdst, t, p, levels, One(), Zero(), backend, allocator)
end
# TODO: braid for `AdjointTensorMap`; think about how to map the `levels` argument.

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
"""
    add_permute!(tdst, tsrc, (p₁, p₂)::Index2Tuple, α::Number, β::Number[, backend])

!!! warning "Deprecated"
    `add_permute!` is deprecated. Use `permute!(tdst, tsrc, p, α, β[, backend])` instead.
"""
function add_permute!(
        tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p::Index2Tuple,
        α::Number, β::Number, backend::AbstractBackend...
    )
    Base.depwarn("`add_permute!` is deprecated, use `permute!` instead", :add_permute!)
    return @inbounds permute!(tdst, tsrc, p, α, β, backend...)
end

"""
    add_braid!(tdst, tsrc, (p₁, p₂)::Index2Tuple, levels::IndexTuple, α::Number, β::Number[, backend])

!!! warning "Deprecated"
    `add_braid!` is deprecated. Use `braid!(tdst, tsrc, p, levels, α, β[, backend])` instead.
"""
function add_braid!(
        tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p::Index2Tuple, levels::IndexTuple,
        α::Number, β::Number, backend::AbstractBackend...
    )
    Base.depwarn("`add_braid!` is deprecated, use `braid!` instead", :add_braid!)
    return @inbounds braid!(tdst, tsrc, p, levels, α, β, backend...)
end

"""
    add_transpose!(tdst, tsrc, (p₁, p₂)::Index2Tuple, α::Number, β::Number[, backend])

!!! warning "Deprecated"
    `add_transpose!` is deprecated. Use `transpose!(tdst, tsrc, p, α, β[, backend])` instead.
"""
function add_transpose!(
        tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p::Index2Tuple,
        α::Number, β::Number, backend::AbstractBackend...
    )
    Base.depwarn("`add_transpose!` is deprecated, use `transpose!` instead", :add_transpose!)
    return @inbounds transpose!(tdst, tsrc, p, α, β, backend...)
end

@propagate_inbounds function add_transform!(
        tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p::Index2Tuple, transformer,
        α::Number, β::Number, backend, allocator
    )
    @boundscheck spacecheck_transform(permute, tdst, tsrc, p)

    if p[1] === codomainind(tsrc) && p[2] === domainind(tsrc)
        add!(tdst, tsrc, α, β)
    else
        I = sectortype(tdst)
        if I === Trivial
            add_trivial_kernel!(tdst, tsrc, p, transformer, α, β, backend, allocator)
        else
            style = FusionStyle(I)
            if use_threaded_transform(tdst, transformer)
                add_kernel_threaded!(style, tdst, tsrc, p, transformer, α, β, backend, allocator)
            else
                add_kernel_nonthreaded!(style, tdst, tsrc, p, transformer, α, β, backend, allocator)
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
function add_trivial_kernel!(tdst, tsrc, p, transformer, α, β, backend, allocator)
    TO.tensoradd!(tdst[], tsrc[], p, false, α, β, backend, allocator)
    return nothing
end

# Non-threaded implementations
# ----------------------------
function add_kernel_nonthreaded!(
        ::UniqueFusion, tdst, tsrc, p, transformer, α, β, backend, allocator
    )
    for (f₁, f₂) in fusiontrees(tsrc)
        _add_transform_single!(tdst, tsrc, p, (f₁, f₂), transformer, α, β, backend, allocator)
    end
    return nothing
end
function add_kernel_nonthreaded!(
        ::UniqueFusion, tdst, tsrc, p, transformer::AbelianTreeTransformer, α, β, backend,
        allocator
    )
    for subtransformer in transformer.data
        _add_transform_single!(tdst, tsrc, p, subtransformer, α, β, backend, allocator)
    end
    return nothing
end
function add_kernel_nonthreaded!(
        ::FusionStyle, tdst, tsrc, p, transformer, α, β, backend, allocator
    )
    # preallocate buffers
    buffers = allocate_buffers(tdst, tsrc, transformer, allocator)

    for src in fusionblocks(tsrc)
        if length(src) == 1
            _add_transform_single!(tdst, tsrc, p, src, transformer, α, β, backend, allocator)
        else
            _add_transform_multi!(tdst, tsrc, p, src, transformer, buffers, α, β, backend, allocator)
        end
    end
    return nothing
end
# specialization in the case of TensorMap
function add_kernel_nonthreaded!(
        ::FusionStyle, tdst, tsrc, p, transformer::GenericTreeTransformer, α, β, backend,
        allocator
    )
    # preallocate buffers
    buffers = allocate_buffers(tdst, tsrc, transformer, allocator)

    for subtransformer in transformer.data
        # Special case without intermediate buffers whenever there is only a single block
        if length(subtransformer[1]) == 1
            _add_transform_single!(tdst, tsrc, p, subtransformer, α, β, backend, allocator)
        else
            _add_transform_multi!(tdst, tsrc, p, subtransformer, buffers, α, β, backend, allocator)
        end
    end
    return nothing
end
# ambiguity resolution
function add_kernel_nonthreaded!(
        ::UniqueFusion, tdst, tsrc, p, transformer::GenericTreeTransformer, α, β, backend,
        allocator
    )
    throw(ArgumentError("Cannot combine `GenericTreeTransformer` with `UniqueFusion`"))
end
# Threaded implementations
# ------------------------
function add_kernel_threaded!(
        ::UniqueFusion, tdst, tsrc, p, transformer, α, β, backend, allocator;
        ntasks::Int = get_num_transformer_threads()
    )
    trees = fusiontrees(tsrc)
    nblocks = length(trees)
    counter = Threads.Atomic{Int}(1)
    Threads.@sync for _ in 1:min(ntasks, nblocks)
        Threads.@spawn begin
            while true
                local_counter = Threads.atomic_add!(counter, 1)
                local_counter > nblocks && break
                @inbounds (f₁, f₂) = trees[local_counter]
                _add_transform_single!(tdst, tsrc, p, (f₁, f₂), transformer, α, β, backend, allocator)
            end
        end
    end
    return nothing
end
function add_kernel_threaded!(
        ::UniqueFusion, tdst, tsrc, p, transformer::AbelianTreeTransformer, α, β, backend,
        allocator; ntasks::Int = get_num_transformer_threads()
    )
    nblocks = length(transformer.data)
    counter = Threads.Atomic{Int}(1)
    Threads.@sync for _ in 1:min(ntasks, nblocks)
        Threads.@spawn begin
            while true
                local_counter = Threads.atomic_add!(counter, 1)
                local_counter > nblocks && break
                @inbounds subtransformer = transformer.data[local_counter]
                _add_transform_single!(tdst, tsrc, p, subtransformer, α, β, backend, allocator)
            end
        end
    end
    return nothing
end

function add_kernel_threaded!(
        ::FusionStyle, tdst, tsrc, p, transformer, α, β, backend, allocator;
        ntasks::Int = get_num_transformer_threads()
    )
    allblocks = fusionblocks(tsrc)
    nblocks = length(allblocks)

    counter = Threads.Atomic{Int}(1)
    Threads.@sync for _ in 1:min(ntasks, nblocks)
        Threads.@spawn begin
            # preallocate buffers for each task
            buffers = allocate_buffers(tdst, tsrc, transformer, allocator)

            while true
                local_counter = Threads.atomic_add!(counter, 1)
                local_counter > nblocks && break
                @inbounds src = allblocks[local_counter]
                if length(src) == 1
                    _add_transform_single!(tdst, tsrc, p, src, transformer, α, β, backend, allocator)
                else
                    _add_transform_multi!(tdst, tsrc, p, src, transformer, buffers, α, β, backend, allocator)
                end
            end
        end
    end

    return nothing
end
# specialization in the case of TensorMap
function add_kernel_threaded!(
        ::FusionStyle, tdst, tsrc, p, transformer::GenericTreeTransformer, α, β, backend,
        allocator; ntasks::Int = get_num_transformer_threads()
    )
    nblocks = length(transformer.data)

    counter = Threads.Atomic{Int}(1)
    Threads.@sync for _ in 1:min(ntasks, nblocks)
        Threads.@spawn begin
            # preallocate buffers for each task
            buffers = allocate_buffers(tdst, tsrc, transformer, allocator)

            while true
                local_counter = Threads.atomic_add!(counter, 1)
                local_counter > nblocks && break
                @inbounds subtransformer = transformer.data[local_counter]
                if length(subtransformer[1]) == 1
                    _add_transform_single!(tdst, tsrc, p, subtransformer, α, β, backend, allocator)
                else
                    _add_transform_multi!(tdst, tsrc, p, subtransformer, buffers, α, β, backend, allocator)
                end
            end
        end
    end

    return nothing
end
# ambiguity resolution
function add_kernel_threaded!(
        ::UniqueFusion, tdst, tsrc, p, transformer::GenericTreeTransformer, α, β, backend,
        allocator; ntasks::Int = get_num_transformer_threads()
    )
    throw(ArgumentError("Cannot combine `GenericTreeTransformer` with `UniqueFusion`"))
end


# Auxiliary methods
# -----------------
function _add_transform_single!(
        tdst, tsrc, p, (f₁, f₂)::FusionTreePair, transformer, α, β, backend, allocator
    )
    (f₁′, f₂′), coeff = transformer((f₁, f₂))
    @inbounds TO.tensoradd!(tdst[f₁′, f₂′], tsrc[f₁, f₂], p, false, α * coeff, β, backend, allocator)
    return nothing
end
function _add_transform_single!(
        tdst, tsrc, p, src::FusionTreeBlock, transformer, α, β, backend, allocator
    )
    dst, U = transformer(src)
    f₁, f₂ = only(fusiontrees(src))
    f₁′, f₂′ = only(fusiontrees(dst))
    coeff = only(U)
    @inbounds TO.tensoradd!(tdst[f₁′, f₂′], tsrc[f₁, f₂], p, false, α * coeff, β, backend, allocator)
    return nothing
end
function _add_transform_single!(
        tdst, tsrc, p, (coeff, struct_dst, struct_src)::AbelianTransformerData,
        α, β, backend, allocator
    )
    subblock_dst = StridedView(tdst.data, struct_dst...)
    subblock_src = StridedView(tsrc.data, struct_src...)
    TO.tensoradd!(subblock_dst, subblock_src, p, false, α * coeff, β, backend, allocator)
    return nothing
end
function _add_transform_single!(
        tdst, tsrc, p, (basistransform, structs_dst, structs_src)::GenericTransformerData,
        α, β, backend, allocator
    )
    struct_dst = (structs_dst[1], only(structs_dst[2])...)
    struct_src = (structs_src[1], only(structs_src[2])...)
    coeff = only(basistransform)
    _add_transform_single!(tdst, tsrc, p, (coeff, struct_dst, struct_src), α, β, backend, allocator)
    return nothing
end

function _add_transform_multi!(
        tdst, tsrc, p, src::FusionTreeBlock, transformer, (buffer1, buffer2), α, β, backend,
        allocator
    )
    dst, U = transformer(src)
    rows, cols = size(U)
    sz_src = size(tsrc[first(fusiontrees(src))...])
    blocksize = prod(sz_src)
    matsize = (
        prod(TupleTools.getindices(sz_src, codomainind(tsrc))),
        prod(TupleTools.getindices(sz_src, domainind(tsrc))),
    )

    # Filling up a buffer with contiguous data
    buffer_src = StridedView(buffer2, (blocksize, cols), (1, blocksize), 0)
    for (i, (f₁, f₂)) in enumerate(fusiontrees(src))
        subblock_src = sreshape(tsrc[f₁, f₂], matsize)
        bufblock_src = sreshape(buffer_src[:, i], matsize)
        copy!(bufblock_src, subblock_src)
    end

    # Resummation into a second buffer using BLAS
    buffer_dst = StridedView(buffer1, (blocksize, rows), (1, blocksize), 0)
    mul!(buffer_dst, buffer_src, transpose(StridedView(U)), α, Zero())

    # Filling up the output
    for (i, (f₃, f₄)) in enumerate(fusiontrees(dst))
        subblock_dst = tdst[f₃, f₄]
        bufblock_dst = sreshape(buffer_dst[:, i], sz_src)
        TO.tensoradd!(subblock_dst, bufblock_dst, p, false, One(), β, backend, allocator)
    end

    return nothing
end
function _add_transform_multi!(
        tdst, tsrc, p, (U, (sz_dst, structs_dst), (sz_src, structs_src)),
        (buffer1, buffer2), α, β, backend, allocator
    )
    rows, cols = size(U)
    blocksize = prod(sz_src)
    matsize = (
        prod(TupleTools.getindices(sz_src, codomainind(tsrc))),
        prod(TupleTools.getindices(sz_src, domainind(tsrc))),
    )

    # Filling up a buffer with contiguous data
    buffer_src = StridedView(buffer2, (blocksize, cols), (1, blocksize), 0)
    for (i, struct_src) in enumerate(structs_src)
        subblock_src = sreshape(StridedView(tsrc.data, sz_src, struct_src...), matsize)
        bufblock_src = sreshape(buffer_src[:, i], matsize)
        copy!(bufblock_src, subblock_src)
    end

    # Resummation into a second buffer using BLAS
    buffer_dst = StridedView(buffer1, (blocksize, rows), (1, blocksize), 0)
    mul!(buffer_dst, buffer_src, transpose(StridedView(U)), α, Zero())

    # Filling up the output
    for (i, struct_dst) in enumerate(structs_dst)
        subblock_dst = StridedView(tdst.data, sz_dst, struct_dst...)
        bufblock_dst = sreshape(buffer_dst[:, i], sz_src)
        TO.tensoradd!(subblock_dst, bufblock_dst, p, false, One(), β, backend, allocator)
    end

    return nothing
end
