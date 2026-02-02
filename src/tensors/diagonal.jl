# DiagonalTensorMap
#==========================================================#
struct DiagonalTensorMap{T, S <: IndexSpace, A <: DenseVector{T}} <: AbstractTensorMap{T, S, 1, 1}
    data::A
    domain::S # equals codomain

    # uninitialized constructors
    function DiagonalTensorMap{T, S, A}(
            ::UndefInitializer, dom::S
        ) where {T, S <: IndexSpace, A <: DenseVector{T}}
        data = A(undef, reduceddim(dom))
        if !isbitstype(T)
            zerovector!(data)
        end
        return DiagonalTensorMap{T, S, A}(data, dom)
    end
    # constructors from data
    function DiagonalTensorMap{T, S, A}(
            data::A, dom::S
        ) where {T, S <: IndexSpace, A <: DenseVector{T}}
        T ⊆ field(S) || @warn("scalartype(data) = $T ⊈ $(field(S)))", maxlog = 1)
        return new{T, S, A}(data, dom)
    end
end

# Basic methods for characterising a tensor:
#--------------------------------------------
space(d::DiagonalTensorMap) = d.domain ← d.domain

storagetype(::Type{<:DiagonalTensorMap{T, S, A}}) where {T, S, A <: DenseVector{T}} = A

# DiagonalTensorMap constructors
#--------------------------------
# undef constructors
"""
    DiagonalTensorMap{T}(undef, domain::S) where {T,S<:IndexSpace}
    # expert mode: select storage type `A`
    DiagonalTensorMap{T,S,A}(undef, domain::S) where {T,S<:IndexSpace,A<:DenseVector{T}}

Construct a `DiagonalTensorMap` with uninitialized data.
"""
function DiagonalTensorMap{T}(::UndefInitializer, V::TensorMapSpace) where {T}
    (numin(V) == numout(V) == 1 && domain(V) == codomain(V)) ||
        throw(ArgumentError("DiagonalTensorMap requires a space with equal domain and codomain and 2 indices"))
    return DiagonalTensorMap{T}(undef, domain(V))
end
function DiagonalTensorMap{T}(::UndefInitializer, V::ProductSpace) where {T}
    length(V) == 1 ||
        throw(ArgumentError("DiagonalTensorMap requires `numin(d) == numout(d) == 1`"))
    return DiagonalTensorMap{T}(undef, only(V))
end
function DiagonalTensorMap{T}(::UndefInitializer, V::S) where {T, S <: IndexSpace}
    return DiagonalTensorMap{T, S, Vector{T}}(undef, V)
end
DiagonalTensorMap(::UndefInitializer, V::IndexSpace) = DiagonalTensorMap{Float64}(undef, V)

function DiagonalTensorMap{T}(data::A, V::S) where {T, S <: IndexSpace, A <: DenseVector{T}}
    length(data) == reduceddim(V) ||
        throw(DimensionMismatch("length(data) = $(length(data)) is not compatible with the space $V"))
    return DiagonalTensorMap{T, S, A}(data, V)
end

function DiagonalTensorMap(data::DenseVector{T}, V::IndexSpace) where {T}
    return DiagonalTensorMap{T}(data, V)
end

function DiagonalTensorMap(t::AbstractTensorMap{T, S, 1, 1}) where {T, S}
    isa(t, DiagonalTensorMap) && return t
    domain(t) == codomain(t) ||
        throw(SpaceMismatch("DiagonalTensorMap requires equal domain and codomain"))
    A = storagetype(t)
    d = DiagonalTensorMap{T, S, A}(undef, space(t, 1))
    for (c, b) in blocks(d)
        bt = block(t, c)
        # TODO: rewrite in terms of `diagview` from MatrixAlgebraKit.jl
        copy!(b.diag, view(bt, LinearAlgebra.diagind(bt)))
    end
    return d
end

Base.similar(d::DiagonalTensorMap) = similar_diagonal(d)
Base.similar(d::DiagonalTensorMap, ::Type{T}) where {T} = similar_diagonal(d, T)

similar_diagonal(d::DiagonalTensorMap) = DiagonalTensorMap(similar(d.data), d.domain)
similar_diagonal(d::DiagonalTensorMap, ::Type{T}) where {T <: Number} =
    DiagonalTensorMap(similar(d.data, T), d.domain)

# TODO: more constructors needed?

# Special case adjoint:
#-----------------------
Base.adjoint(d::DiagonalTensorMap{<:Real}) = d
Base.adjoint(d::DiagonalTensorMap{<:Complex}) = DiagonalTensorMap(conj(d.data), d.domain)

# Efficient copy constructors and TensorMap converters
#-----------------------------------------------------
Base.copy(d::DiagonalTensorMap) = typeof(d)(copy(d.data), d.domain)

function Base.copy!(t::AbstractTensorMap, d::DiagonalTensorMap)
    space(t) == space(d) || throw(SpaceMismatch())
    for (c, b) in blocks(d)
        copy!(block(t, c), b)
    end
    return t
end
TensorMap(d::DiagonalTensorMap) = copy!(similar(d, scalartype(d), space(d)), d)
Base.convert(::Type{TensorMap}, d::DiagonalTensorMap) = TensorMap(d)

function Base.convert(D::Type{<:DiagonalTensorMap}, d::DiagonalTensorMap)
    return (d isa D) ? d : DiagonalTensorMap(convert(storagetype(D), d.data), d.domain)
end
Base.convert(::Type{DiagonalTensorMap}, t::DiagonalTensorMap) = t
function Base.convert(::Type{DiagonalTensorMap}, t::AbstractTensorMap)
    LinearAlgebra.isdiag(t) ||
        throw(ArgumentError("DiagonalTensorMap requires input tensor that is diagonal"))
    return DiagonalTensorMap(t)
end
function Base.convert(::Type{DiagonalTensorMap}, d::Dict{Symbol, Any})
    return convert(DiagonalTensorMap, convert(TensorMap, d))
end

# Complex, real and imaginary parts
#-----------------------------------
for f in (:real, :imag, :complex)
    @eval begin
        function Base.$f(d::DiagonalTensorMap)
            return DiagonalTensorMap($f(d.data), d.domain)
        end
    end
end

# Getting and setting the data at the block level
#-------------------------------------------------
function block(d::DiagonalTensorMap, s::Sector)
    sectortype(d) == typeof(s) || throw(SectorMismatch())
    offset = 0
    dom = domain(d)[1]
    for c in blocksectors(d)
        if c < s
            offset += dim(dom, c)
        elseif c == s
            r = offset .+ (1:dim(dom, c))
            return Diagonal(view(d.data, r))
        else # s not in sectors(t)
            break
        end
    end
    return Diagonal(view(d.data, 1:0))
end

blocks(t::DiagonalTensorMap) = BlockIterator(t, diagonalblockstructure(space(t)))
function blocktype(::Type{DiagonalTensorMap{T, S, A}}) where {T, S, A}
    return Diagonal{T, SubArray{T, 1, A, Tuple{UnitRange{Int}}, true}}
end

function Base.iterate(iter::BlockIterator{<:DiagonalTensorMap}, state...)
    next = iterate(iter.structure, state...)
    isnothing(next) && return next
    (c, r), newstate = next
    return c => Diagonal(view(iter.t.data, r)), newstate
end

function Base.getindex(iter::BlockIterator{<:DiagonalTensorMap}, c::Sector)
    sectortype(iter.t) === typeof(c) || throw(SectorMismatch())
    r = get(iter.structure, c, 1:0)
    return Diagonal(view(iter.t.data, r))
end

# Indexing and getting and setting the data at the subblock level
#-----------------------------------------------------------------
Base.@propagate_inbounds function subblock(
        d::DiagonalTensorMap, (f₁, f₂)::Tuple{FusionTree{I, 1}, FusionTree{I, 1}}
    ) where {I <: Sector}
    s = f₁.uncoupled[1]
    s == f₁.coupled == f₂.uncoupled[1] == f₂.coupled || throw(SectorMismatch())
    return block(d, s)
end

function Base.getindex(d::DiagonalTensorMap)
    sectortype(d) === Trivial || throw(SectorMismatch())
    return Diagonal(d.data)
end

# Index manipulations
# -------------------
function has_shared_permute(d::DiagonalTensorMap, (p₁, p₂)::Index2Tuple)
    if p₁ === (1,) && p₂ === (2,)
        return true
    elseif p₁ === (2,) && p₂ === (1,) # transpose
        return sectortype(d) === Trivial
    else
        return false
    end
end

function permute(
        d::DiagonalTensorMap, (p₁, p₂)::Index2Tuple{1, 1}; copy::Bool = false
    )
    if p₁ === (1,) && p₂ === (2,)
        return copy ? Base.copy(d) : d
    elseif p₁ === (2,) && p₂ === (1,) # transpose
        if has_shared_permute(d, (p₁, p₂)) # tranpose for bosonic sectors
            return DiagonalTensorMap(copy ? Base.copy(d.data) : d.data, dual(d.domain))
        end
        d′ = typeof(d)(undef, dual(d.domain))
        for (c, b) in blocks(d)
            f = only(fusiontrees(codomain(d), c))
            ((f′, _), coeff) = only(permute(f, f, p₁, p₂))
            c′ = f′.coupled
            scale!(block(d′, c′), b, coeff)
        end
        return d′
    else
        throw(ArgumentError(lazy"invalid permutation $((p₁, p₂)) for tensor in space $(space(d))"))
    end
end

function LinearAlgebra.transpose(
        d::DiagonalTensorMap, (p₁, p₂)::Index2Tuple{1, 1}; copy::Bool = false
    )
    if p₁ === (1,) && p₂ === (2,)
        return copy ? Base.copy(d) : d
    elseif p₁ === (2,) && p₂ === (1,) # transpose
        if has_shared_permute(d, (p₁, p₂)) # tranpose for bosonic sectors
            return DiagonalTensorMap(copy ? Base.copy(d.data) : d.data, dual(d.domain))
        end
        d′ = typeof(d)(undef, dual(d.domain))
        for (c, b) in blocks(d)
            f = only(fusiontrees(codomain(d), c))
            ((f′, _), coeff) = only(transpose(f, f, p₁, p₂))
            c′ = f′.coupled
            scale!(block(d′, c′), b, coeff)
        end
        return d′
    else
        throw(ArgumentError(lazy"invalid transposition $((p₁, p₂)) for tensor in space $(space(d))"))
    end
end

# VectorInterface
# ---------------
function VectorInterface.zerovector(d::DiagonalTensorMap, ::Type{S}) where {S <: Number}
    return DiagonalTensorMap(zerovector(d.data, S), d.domain)
end
function VectorInterface.add(ty::DiagonalTensorMap, tx::DiagonalTensorMap, α::Number, β::Number)
    domain(ty) == domain(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    T = VectorInterface.promote_add(ty, tx, α, β)
    return add!(scale!(zerovector(ty, T), ty, β), tx, α) # zerovector instead of similar preserves diagonal structure
end

# TensorOperations
# ----------------
function TO.tensoradd_type(TC, A::DiagonalTensorMap, ::Index2Tuple{1, 1}, ::Bool)
    M = similarstoragetype(A, TC)
    return DiagonalTensorMap{TC, spacetype(A), M}
end

function TO.tensorcontract_type(
        TC, A::DiagonalTensorMap, ::Index2Tuple{1, 1}, ::Bool,
        B::DiagonalTensorMap, ::Index2Tuple{1, 1}, ::Bool,
        ::Index2Tuple{1, 1}
    )
    S = check_spacetype(A, B)
    TC′ = promote_permute(TC, sectortype(S))
    M = promote_storagetype(similarstoragetype(A, TC′), similarstoragetype(B, TC′))
    return DiagonalTensorMap{TC, S, M}
end

function TO.tensoralloc(
        ::Type{DiagonalTensorMap{T, S, M}}, structure::TensorMapSpace{S, 1, 1},
        istemp::Val, allocator = TO.DefaultAllocator()
    ) where {T, S, M}
    domain(structure) == codomain(structure) || throw(ArgumentError("domain ≠ codomain"))
    V = only(domain(structure))
    dim = reduceddim(V)
    data = TO.tensoralloc(M, dim, istemp, allocator)
    return DiagonalTensorMap{T, S, M}(data, V)
end

# Linear Algebra and factorizations
# ---------------------------------
function one!(d::DiagonalTensorMap)
    fill!(d.data, one(eltype(d.data)))
    return d
end
function Base.one(d::DiagonalTensorMap)
    return DiagonalTensorMap(one.(d.data), d.domain)
end
function Base.zero(d::DiagonalTensorMap)
    return DiagonalTensorMap(zero.(d.data), d.domain)
end

function compose_dest(A::DiagonalTensorMap, B::DiagonalTensorMap)
    S = check_spacetype(A, B)
    TC = TO.promote_contract(scalartype(A), scalartype(B), One)
    M = promote_storagetype(similarstoragetype(A, TC), similarstoragetype(B, TC))
    TTC = DiagonalTensorMap{TC, S, M}
    structure = codomain(A) ← domain(B)
    return TO.tensoralloc(TTC, structure, Val(false))
end

function LinearAlgebra.mul!(
        dC::DiagonalTensorMap, dA::DiagonalTensorMap, dB::DiagonalTensorMap, α::Number, β::Number
    )
    dC.domain == dA.domain == dB.domain || throw(SpaceMismatch())
    mul!(Diagonal(dC.data), Diagonal(dA.data), Diagonal(dB.data), α, β)
    return dC
end

function LinearAlgebra.lmul!(D::DiagonalTensorMap, t::AbstractTensorMap)
    domain(D) == codomain(t) || throw(SpaceMismatch())
    foreachblock(D, t) do c, bs
        return lmul!(bs...)
    end
    return t
end

function LinearAlgebra.rmul!(t::AbstractTensorMap, D::DiagonalTensorMap)
    codomain(D) == domain(t) || throw(SpaceMismatch())
    foreachblock(t, D) do c, bs
        return rmul!(bs...)
    end
    return t
end

Base.inv(d::DiagonalTensorMap) = DiagonalTensorMap(inv.(d.data), d.domain)
function Base.:\(d1::DiagonalTensorMap, d2::DiagonalTensorMap)
    d1.domain == d2.domain || throw(SpaceMismatch())
    return DiagonalTensorMap(d1.data .\ d2.data, d1.domain)
end
function Base.:/(d1::DiagonalTensorMap, d2::DiagonalTensorMap)
    d1.domain == d2.domain || throw(SpaceMismatch())
    return DiagonalTensorMap(d1.data ./ d2.data, d1.domain)
end
function LinearAlgebra.pinv(d::DiagonalTensorMap; kwargs...)
    T = eltype(d.data)
    atol = get(kwargs, :atol, zero(real(T)))
    if iszero(atol)
        rtol = get(kwargs, :rtol, zero(real(T)))
    else
        rtol = sqrt(eps(real(float(one(T))))) * length(d.data)
    end
    pdata = let tol = max(atol, rtol * maximum(abs, d.data))
        map(x -> abs(x) < tol ? zero(x) : pinv(x), d.data)
    end
    return DiagonalTensorMap(pdata, d.domain)
end
function LinearAlgebra.isposdef(d::DiagonalTensorMap)
    return all(isposdef, d.data)
end

# matrix functions
for f in
    (
        :exp, :cos, :sin, :tan, :cot, :cosh, :sinh, :tanh, :coth, :atan, :acot, :asinh, :sqrt,
        :log, :asin, :acos, :acosh, :atanh, :acoth,
    )
    @eval Base.$f(d::DiagonalTensorMap) = DiagonalTensorMap($f.(d.data), d.domain)
end

# Show
#------
function type_repr(::Type{DiagonalTensorMap{T, S, A}}) where {T, S, A}
    return "DiagonalTensorMap{$T, $(type_repr(S)), $A}"
end
function Base.showarg(io::IO, t::DiagonalTensorMap, toplevel::Bool)
    !toplevel && print(io, "::")
    print(io, type_repr(typeof(t)))
    return nothing
end
Base.show(io::IO, t::DiagonalTensorMap) =
    print(io, type_repr(typeof(t)), "(", t.data, ", ", space(t, 1), ")")
