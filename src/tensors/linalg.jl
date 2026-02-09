# Basic algebra
#---------------
Base.copy(t::AbstractTensorMap) = Base.copy!(similar(t), t)

Base.:-(t::AbstractTensorMap) = VectorInterface.scale(t, -one(scalartype(t)))

Base.:+(t1::AbstractTensorMap, t2::AbstractTensorMap) = VectorInterface.add(t1, t2)
function Base.:-(t1::AbstractTensorMap, t2::AbstractTensorMap)
    return VectorInterface.add(t1, t2, -one(scalartype(t1)))
end

Base.:*(t::AbstractTensorMap, α::Number) = VectorInterface.scale(t, α)
Base.:*(α::Number, t::AbstractTensorMap) = VectorInterface.scale(t, α)

Base.:/(t::AbstractTensorMap, α::Number) = *(t, one(scalartype(t)) / α)
Base.:\(α::Number, t::AbstractTensorMap) = *(t, one(scalartype(t)) / α)

LinearAlgebra.normalize!(t::AbstractTensorMap, p::Real = 2) = scale!(t, inv(norm(t, p)))
LinearAlgebra.normalize(t::AbstractTensorMap, p::Real = 2) = scale(t, inv(norm(t, p)))

# destination allocation for matrix multiplication
# note that we don't fall back to `tensoralloc_contract` since that needs to account for
# permutations, which might require complex scalartypes even if the inputs are real.
function compose_dest(A::AbstractTensorMap, B::AbstractTensorMap)
    S = check_spacetype(A, B)
    M = promote_storagetype(TO.promote_contract(scalartype(A), scalartype(B), One), A, B)
    TTC = tensormaptype(S, numout(A), numin(B), M)
    structure = codomain(A) ← domain(B)
    return TO.tensoralloc(TTC, structure, Val(false))
end

"""
    compose(t1::AbstractTensorMap, t2::AbstractTensorMap) -> AbstractTensorMap

Return the `AbstractTensorMap` that implements the composition of the two tensor maps `t1`
and `t2`.
"""
function compose(A::AbstractTensorMap, B::AbstractTensorMap)
    C = compose_dest(A, B)
    return mul!(C, A, B)
end
Base.:*(t1::AbstractTensorMap, t2::AbstractTensorMap) = compose(t1, t2)

Base.exp(t::AbstractTensorMap) = exp!(copy(t))
function Base.:^(t::AbstractTensorMap, p::Integer)
    return p < 0 ? Base.power_by_squaring(inv(t), -p) : Base.power_by_squaring(t, p)
end

# Special purpose constructors
#------------------------------
Base.zero(t::AbstractTensorMap) = VectorInterface.zerovector(t)
function Base.one(t::AbstractTensorMap)
    domain(t) == codomain(t) ||
        throw(SectorMismatch("no identity if domain and codomain are different"))
    return one!(similar(t))
end
function one!(t::AbstractTensorMap)
    domain(t) == codomain(t) ||
        throw(SectorMismatch("no identity if domain and codomain are different"))
    for (c, b) in blocks(t)
        one!(b)
    end
    return t
end

@doc """
    id([T::Type=Float64,] V::TensorSpace) -> TensorMap
    id!(t::AbstractTensorMap) -> AbstractTensorMap

Construct the identity endomorphism on space `V`, i.e. return a `t::TensorMap` with
`domain(t) == codomain(t) == V`, where either `scalartype(t) = T` if `T` is a `Number` type
or `storagetype(t) = T` if `T` is a `DenseVector` type.

See also [`one!`](@ref).
""" id, id!

id(V::TensorSpace) = id(Float64, V)
function id(A::Type, V::TensorSpace{S}) where {S}
    W = V ← V
    N = length(codomain(W))
    dst = tensormaptype(S, N, N, A)(undef, W)
    return id!(dst)
end
const id! = one!

@doc """
    isomorphism([T::Type=Float64,] codomain::TensorSpace, domain::TensorSpace) -> TensorMap
    isomorphism([T::Type=Float64,] codomain ← domain) -> TensorMap
    isomorphism([T::Type=Float64,] domain → codomain) -> TensorMap
    isomorphism!(t::AbstractTensorMap) -> AbstractTensorMap

Construct a specific isomorphism between the codomain and the domain, i.e. return a
`t::TensorMap` where either `scalartype(t) = T` if `T` is a `Number` type or
`storagetype(t) = T` if `T` is a `DenseVector` type. If the spaces are not isomorphic, an
error will be thrown.

!!! note
    There is no canonical choice for a specific isomorphism, but the current choice is such
    that `isomorphism(cod, dom) == inv(isomorphism(dom, cod))`.

See also [`unitary`](@ref) when `InnerProductStyle(cod) === EuclideanInnerProduct()`.
""" isomorphism, isomorphism!

function isomorphism!(t::AbstractTensorMap)
    domain(t) ≅ codomain(t) ||
        throw(SpaceMismatch(lazy"domain and codomain are not isomorphic: $(space(t))"))
    for (_, b) in blocks(t)
        one!(b)
    end
    return t
end

@doc """
    unitary([T::Type=Float64,] codomain::TensorSpace, domain::TensorSpace) -> TensorMap
    unitary([T::Type=Float64,] codomain ← domain) -> TensorMap
    unitary([T::Type=Float64,] domain → codomain) -> TensorMap
    unitary!(t::AbstractTensorMap) -> AbstractTensorMap

Construct a specific unitary morphism between the codomain and the domain, i.e. return a
`t::TensorMap` where either `scalartype(t) = T` if `T` is a `Number` type or
`storagetype(t) = T` if `T` is a `DenseVector` type. If the spaces are not isomorphic, or
the spacetype does not have a Euclidean inner product, an error will be thrown.

!!! note
    There is no canonical choice for a specific unitary, but the current choice is such that
    `unitary(cod, dom) == inv(unitary(dom, cod)) = adjoint(unitary(dom, cod))`.

See also [`isomorphism`](@ref) and [`isometry`](@ref).
""" unitary, unitary!

function unitary!(t::AbstractTensorMap)
    InnerProductStyle(t) === EuclideanInnerProduct() || throw_invalid_innerproduct(:unitary)
    return isomorphism!(t)
end

@doc """
    isometry([T::Type=Float64,] codomain::TensorSpace, domain::TensorSpace) -> TensorMap
    isometry([T::Type=Float64,] codomain ← domain) -> TensorMap
    isometry([T::Type=Float64,] domain → codomain) -> TensorMap
    isometry!(t::AbstractTensorMap) -> AbstractTensorMap

Construct a specific isometry between the codomain and the domain, i.e. return a
`t::TensorMap` where either `scalartype(t) = T` if `T` is a `Number` type or
`storagetype(t) = T` if `T` is a `DenseVector` type. The isometry `t` then satisfies
`t' * t = id(domain)` and `(t * t')^2 = t * t'`. If the spaces do not allow for such an 
isometric inclusion, an error will be thrown.

See also [`isomorphism`](@ref) and [`unitary`](@ref).
""" isometry, isometry!

function isometry!(t::AbstractTensorMap)
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:isometry)
    domain(t) ≾ codomain(t) ||
        throw(SpaceMismatch(lazy"domain and codomain are not isometrically embeddable: $(space(t))"))
    for (_, b) in blocks(t)
        one!(b)
    end
    return t
end

# expand methods with default arguments
for morphism in (:isomorphism, :unitary, :isometry)
    morphism! = Symbol(morphism, :!)
    @eval begin
        $morphism(V::TensorMapSpace) = $morphism(Float64, V)
        $morphism(codomain::TensorSpace, domain::TensorSpace) = $morphism(codomain ← domain)
        function $morphism(A::Type, codomain::TensorSpace, domain::TensorSpace)
            return $morphism(A, codomain ← domain)
        end
        function $morphism(A::Type, V::TensorMapSpace{S, N₁, N₂}) where {S, N₁, N₂}
            t = tensormaptype(S, N₁, N₂, A)(undef, V)
            return $morphism!(t)
        end
        $morphism(t::AbstractTensorMap) = $morphism!(similar(t))
    end
end

# Diagonal tensors
# ----------------
function LinearAlgebra.diag(t::AbstractTensorMap)
    return SectorDict(c => LinearAlgebra.diag(b) for (c, b) in blocks(t))
end
function LinearAlgebra.diagm(codom::VectorSpace, dom::VectorSpace, v::SectorDict)
    return TensorMap(
        SectorDict(
            c => LinearAlgebra.diagm(blockdim(codom, c), blockdim(dom, c), b)
                for (c, b) in v
        ), codom ← dom
    )
end
LinearAlgebra.isdiag(t::AbstractTensorMap) = all(LinearAlgebra.isdiag ∘ last, blocks(t))

# In-place methods
#------------------

# Copy, adjoint and fill:
function Base.copy!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap)
    space(tdst) == space(tsrc) || throw(SpaceMismatch("$(space(tdst)) ≠ $(space(tsrc))"))
    for ((c, bdst), (_, bsrc)) in zip(blocks(tdst), blocks(tsrc))
        copy!(bdst, bsrc)
    end
    return tdst
end
function Base.copy!(tdst::TensorMap, tsrc::TensorMap)
    space(tdst) == space(tsrc) || throw(SpaceMismatch("$(space(tdst)) ≠ $(space(tsrc))"))
    copy!(tdst.data, tsrc.data)
    return tdst
end
function Base.fill!(t::AbstractTensorMap, value::Number)
    for (c, b) in blocks(t)
        fill!(b, value)
    end
    return t
end
function Base.fill!(t::TensorMap, value::Number)
    fill!(t.data, value)
    return t
end
function LinearAlgebra.adjoint!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap)
    InnerProductStyle(tdst) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:adjoint!)
    space(tdst) == adjoint(space(tsrc)) ||
        throw(SpaceMismatch("$(space(tdst)) ≠ adjoint($(space(tsrc)))"))
    for c in blocksectors(tdst)
        adjoint!(StridedView(block(tdst, c)), StridedView(block(tsrc, c)))
    end
    return tdst
end

# Basic vector space methods: recycle VectorInterface implementation
function LinearAlgebra.rmul!(t::AbstractTensorMap, α::Number)
    return iszero(α) ? zerovector!(t) : scale!(t, α)
end
function LinearAlgebra.lmul!(α::Number, t::AbstractTensorMap)
    return iszero(α) ? zerovector!(t) : scale!(t, α)
end

function LinearAlgebra.mul!(t1::AbstractTensorMap, t2::AbstractTensorMap, α::Number)
    return scale!(t1, t2, α)
end
function LinearAlgebra.mul!(t1::AbstractTensorMap, α::Number, t2::AbstractTensorMap)
    return scale!(t1, t2, α)
end

# TODO: remove VectorInterface namespace when we renamed TensorKit.add!
function LinearAlgebra.axpy!(α::Number, t1::AbstractTensorMap, t2::AbstractTensorMap)
    return VectorInterface.add!(t2, t1, α)
end
function LinearAlgebra.axpby!(
        α::Number, t1::AbstractTensorMap, β::Number, t2::AbstractTensorMap
    )
    return VectorInterface.add!(t2, t1, α, β)
end

# inner product and norm only valid for spaces with Euclidean inner product
LinearAlgebra.dot(t1::AbstractTensorMap, t2::AbstractTensorMap) = inner(t1, t2)

function LinearAlgebra.norm(t::AbstractTensorMap, p::Real = 2)
    InnerProductStyle(t) === EuclideanInnerProduct() || throw_invalid_innerproduct(:norm)
    return _norm(blocks(t), p, float(zero(real(scalartype(t)))))
end
function _norm(blockiter, p::Real, init::Real)
    if p == Inf
        return mapreduce(max, blockiter; init = init) do (c, b)
            return isempty(b) ? init : oftype(init, LinearAlgebra.normInf(b))
        end
    elseif p > 0 # finite positive p
        np = init
        for (c, b) in blockiter
            np += oftype(init, dim(c) * norm(b, p)^p)
        end
        return np^(inv(oftype(np, p)))
    else
        msg = "Norm with non-positive p is not defined for `AbstractTensorMap`"
        throw(ArgumentError(msg))
    end
end
function LinearAlgebra.norm(t::TensorMap, p::Real = 2)
    InnerProductStyle(t) === EuclideanInnerProduct() || throw_invalid_innerproduct(:norm)
    # performance specialization:
    FusionStyle(sectortype(t)) isa UniqueFusion && return norm(t.data, p)
    return _norm(blocks(t), p, float(zero(real(scalartype(t)))))
end

_default_rtol(t) = eps(real(float(scalartype(t)))) * min(dim(domain(t)), dim(codomain(t)))

function LinearAlgebra.rank(
        t::AbstractTensorMap;
        atol::Real = 0, rtol::Real = atol > 0 ? 0 : _default_rtol(t)
    )
    r = zero(dimscalartype(sectortype(t)))
    iszero(dim(t)) && return r
    S = MatrixAlgebraKit.svd_vals(t)
    tol = max(atol, rtol * maximum(parent(S)))
    for (c, b) in pairs(S)
        if !isempty(b)
            r += dim(c) * count(>(tol), b)
        end
    end
    return r
end

function LinearAlgebra.cond(t::AbstractTensorMap, p::Real = 2)
    if p == 2
        if dim(t) == 0
            domain(t) == codomain(t) ||
                throw(SpaceMismatch("`cond` requires domain and codomain to be the same"))
            return zero(real(float(scalartype(t))))
        end
        S = MatrixAlgebraKit.svd_vals(t)
        maxS = maximum(parent(S))
        minS = minimum(parent(S))
        return iszero(maxS) ? oftype(maxS, Inf) : (maxS / minS)
    else
        throw(ArgumentError("cond currently only defined for p=2"))
    end
end

# TensorMap trace
function LinearAlgebra.tr(t::AbstractTensorMap)
    domain(t) == codomain(t) ||
        throw(SpaceMismatch("Trace of a tensor only exist when domain == codomain"))
    s = zero(scalartype(t)) * zero(dimscalartype(sectortype(t)))
    for (c, b) in blocks(t)
        s += dim(c) * tr(b)
    end
    return s
end

# TensorMap multiplication
function LinearAlgebra.mul!(
        tC::AbstractTensorMap, tA::AbstractTensorMap, tB::AbstractTensorMap, α = true, β = false
    )
    compose(space(tA), space(tB)) == space(tC) ||
        throw(SpaceMismatch(lazy"$(space(tC)) ≠ $(space(tA)) * $(space(tB))"))

    iterC = blocks(tC)
    iterA = blocks(tA)
    iterB = blocks(tB)
    nextA = iterate(iterA)
    nextB = iterate(iterB)
    nextC = iterate(iterC)
    while !isnothing(nextC)
        (cC, C), stateC = nextC
        if !isnothing(nextA) && !isnothing(nextB)
            (cA, A), stateA = nextA
            (cB, B), stateB = nextB
            if cA == cC && cB == cC
                mul!(C, A, B, α, β)
                nextA = iterate(iterA, stateA)
                nextB = iterate(iterB, stateB)
                nextC = iterate(iterC, stateC)
            elseif cA < cC
                nextA = iterate(iterA, stateA)
            elseif cB < cC
                nextB = iterate(iterB, stateB)
            else
                if β != one(β)
                    rmul!(C, β)
                end
                nextC = iterate(iterC, stateC)
            end
        else
            if β != one(β)
                rmul!(C, β)
            end
            nextC = iterate(iterC, stateC)
        end
    end
    return tC
end

# TODO: consider spawning threads for different blocks, support backends

# TensorMap inverse
function Base.inv(t::AbstractTensorMap)
    cod = codomain(t)
    dom = domain(t)
    cod ≅ dom ||
        throw(SpaceMismatch("codomain $cod and domain $dom are not isomorphic: no inverse"))
    T = float(scalartype(t))
    tinv = similar(t, T, dom ← cod)
    for (c, b) in blocks(t)
        binv = one!(block(tinv, c))
        ldiv!(lu(b), binv)
    end
    return tinv
end
function LinearAlgebra.pinv(t::AbstractTensorMap; kwargs...)
    T = float(scalartype(t))
    tpinv = similar(t, T, domain(t) ← codomain(t))
    # TODO: fix so that `rtol` used total tensor norm instead of per block
    for (c, b) in blocks(t)
        copy!(block(tpinv, c), pinv(b; kwargs...))
    end
    return tpinv
end
function Base.:(\)(t1::AbstractTensorMap, t2::AbstractTensorMap)
    codomain(t1) == codomain(t2) ||
        throw(SpaceMismatch("non-matching codomains in t1 \\ t2"))

    T = float(promote_type(scalartype(t1), scalartype(t2)))
    t = similar(t1, T, domain(t1) ← domain(t2))
    for (c, b) in blocks(t)
        copy!(b, block(t1, c) \ block(t2, c))
    end
    return t
end
function Base.:(/)(t1::AbstractTensorMap, t2::AbstractTensorMap)
    domain(t1) == domain(t2) ||
        throw(SpaceMismatch("non-matching domains in t1 / t2"))
    T = promote_type(scalartype(t1), scalartype(t2))
    t = similar(t1, T, codomain(t1) ← codomain(t2))
    for (c, b) in blocks(t)
        copy!(b, block(t1, c) / block(t2, c))
    end
    return t
end

# TensorMap exponentation:
function exp!(t::TensorMap)
    domain(t) == codomain(t) ||
        error("Exponential of a tensor only exist when domain == codomain.")
    for (c, b) in blocks(t)
        copy!(b, LinearAlgebra.exp!(b))
    end
    return t
end

# Sylvester equation with TensorMap objects:
function LinearAlgebra.sylvester(A::AbstractTensorMap, B::AbstractTensorMap, C::AbstractTensorMap)
    (codomain(A) == domain(A) == codomain(C) && codomain(B) == domain(B) == domain(C)) ||
        throw(SpaceMismatch())
    cod = domain(A)
    dom = codomain(B)
    T = float(promote_type(scalartype(A), scalartype(B), scalartype(C)))
    t = similar(C, T, cod ← dom)
    for (c, b) in blocks(t)
        copy!(b, sylvester(block(A, c), block(B, c), block(C, c)))
    end
    return t
end

# functions that map ℝ to (a subset of) ℝ
for f in (:cos, :sin, :tan, :cot, :cosh, :sinh, :tanh, :coth, :atan, :acot, :asinh)
    sf = string(f)
    @eval function Base.$f(t::AbstractTensorMap)
        domain(t) == codomain(t) ||
            throw(SpaceMismatch("`$($sf)` of a tensor only exist when domain == codomain"))
        T = float(scalartype(t))
        tf = similar(t, T)
        if T <: Real
            for (c, b) in blocks(t)
                copy!(block(tf, c), real(MatrixAlgebraKit.$f(b)))
            end
        else
            for (c, b) in blocks(t)
                copy!(block(tf, c), MatrixAlgebraKit.$f(b))
            end
        end
        return tf
    end
end
# functions that don't map ℝ to (a subset of) ℝ
for f in (:sqrt, :log, :asin, :acos, :acosh, :atanh, :acoth)
    sf = string(f)
    @eval function Base.$f(t::AbstractTensorMap)
        domain(t) == codomain(t) ||
            throw(SpaceMismatch("`$($sf)` of a tensor only exist when domain == codomain"))
        T = complex(float(scalartype(t)))
        tf = similar(t, T)
        for (c, b) in blocks(t)
            copy!(block(tf, c), $f(b))
        end
        return tf
    end
end

# concatenate tensors
function catdomain(t1::TT, t2::TT) where {S, N₁, TT <: AbstractTensorMap{<:Any, S, N₁, 1}}
    codomain(t1) == codomain(t2) ||
        throw(
        SpaceMismatch("codomains of tensors to concatenate must match:\n$(codomain(t1)) ≠ $(codomain(t2))")
    )
    V1, = domain(t1)
    V2, = domain(t2)
    isdual(V1) == isdual(V2) ||
        throw(SpaceMismatch("cannot horizontally concatenate tensors whose domain has non-matching duality"))

    V = V1 ⊕ V2
    T = promote_type(scalartype(t1), scalartype(t2))
    t = similar(t1, T, codomain(t1) ← V)
    for (c, b) in blocks(t)
        b[:, 1:dim(V1, c)] .= block(t1, c)
        b[:, dim(V1, c) .+ (1:dim(V2, c))] .= block(t2, c)
    end
    return t
end
function catcodomain(t1::TT, t2::TT) where {S, N₂, TT <: AbstractTensorMap{<:Any, S, 1, N₂}}
    domain(t1) == domain(t2) ||
        throw(SpaceMismatch("domains of tensors to concatenate must match:\n$(domain(t1)) ≠ $(domain(t2))"))
    V1, = codomain(t1)
    V2, = codomain(t2)
    isdual(V1) == isdual(V2) ||
        throw(SpaceMismatch("cannot vertically concatenate tensors whose codomain has non-matching duality"))

    V = V1 ⊕ V2
    T = promote_type(scalartype(t1), scalartype(t2))
    t = similar(t1, T, V ← domain(t1))
    for (c, b) in blocks(t)
        b[1:dim(V1, c), :] .= block(t1, c)
        b[dim(V1, c) .+ (1:dim(V2, c)), :] .= block(t2, c)
    end
    return t
end

"""
    absorb(tdst::AbstractTensorMap, tsrc::AbstractTensorMap)
    absorb!(tdst::AbstactTensorMap, tsrc::AbstractTensorMap)

Absorb the contents of `tsrc` into `tdst`, which may have different sizes of data.
This is equivalent to the following operation on dense arrays, but also works for symmetric
tensors. Note also that this only overwrites the regions that are shared, and will do
nothing on the ones that are not, so it is up to the user to properly initialize the
destination.

```julia
sub_axes = map((x, y) -> 1:min(x, y), size(tdst), size(tsrc))
tdst[sub_axes...] .= tsrc[sub_axes...]
```
"""
absorb(tdst::AbstractTensorMap, tsrc::AbstractTensorMap) = absorb!(copy(tdst), tsrc)
function absorb!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap)
    numin(tdst) == numin(tsrc) && numout(tdst) == numout(tsrc) ||
        throw(DimensionError("Incompatible number of indices for source and destination"))
    S = check_spacetype(tdst, tsrc)
    dom = mapreduce(infimum, ⊗, domain(tdst), domain(tsrc); init = one(S))
    cod = mapreduce(infimum, ⊗, codomain(tdst), codomain(tsrc); init = one(S))
    for (f1, f2) in fusiontrees(cod ← dom)
        @inbounds data_dst = tdst[f1, f2]
        @inbounds data_src = tsrc[f1, f2]
        sub_axes = map(Base.OneTo ∘ min, size(data_dst), size(data_src))
        data_dst[sub_axes...] .= data_src[sub_axes...]
    end
    return tdst
end

# tensor product of tensors
"""
    ⊗(t1::AbstractTensorMap, t2::AbstractTensorMap, ...) -> TensorMap
    otimes(t1::AbstractTensorMap, t2::AbstractTensorMap, ...) -> TensorMap

Compute the tensor product between two `AbstractTensorMap` instances, which results in a
new `TensorMap` instance whose codomain is `codomain(t1) ⊗ codomain(t2)` and whose domain
is `domain(t1) ⊗ domain(t2)`.
"""
function ⊗(A::AbstractTensorMap, B::AbstractTensorMap)
    check_spacetype(A, B)

    # allocate destination with correct scalartype
    pA = ((codomainind(A)..., domainind(A)...), ())
    pB = ((), (codomainind(B)..., domainind(B)...))
    NA = numind(A)
    pAB = (
        (codomainind(A)..., (codomainind(B) .+ NA)...),
        (domainind(A)..., (domainind(B) .+ NA)...),
    )
    TC = TO.promote_contract(scalartype(A), scalartype(B))
    C = TO.tensoralloc_contract(TC, A, pA, false, B, pB, false, pAB, Val(false))
    zerovector!(C)

    # implement tensor product
    for (f1l, f1r) in fusiontrees(A)
        @inbounds a = A[f1l, f1r]
        for (f2l, f2r) in fusiontrees(B)
            @inbounds b = B[f2l, f2r]
            c1 = f1l.coupled # = f1r.coupled
            c2 = f2l.coupled # = f2r.coupled
            for c in c1 ⊗ c2, μ in 1:Nsymbol(c1, c2, c)
                for (fl, coeff1) in merge(f1l, f2l, c, μ)
                    for (fr, coeff2) in merge(f1r, f2r, c, μ)
                        TO.tensorcontract!(
                            C[fl, fr],
                            A[f1l, f1r], pA, false,
                            B[f2l, f2r], pB, false,
                            pAB,
                            coeff1 * conj(coeff2), One()
                        )
                    end
                end
            end
        end
    end
    return C
end

# deligne product of tensors
function ⊠(t1::AbstractTensorMap, t2::AbstractTensorMap)
    S1 = spacetype(t1)
    I1 = sectortype(S1)
    S2 = spacetype(t2)
    I2 = sectortype(S2)
    codom1 = codomain(t1) ⊠ one(S2)
    dom1 = domain(t1) ⊠ one(S2)
    t1′ = similar(t1, codom1 ← dom1)
    for (c, b) in blocks(t1)
        copy!(block(t1′, c ⊠ unit(I2)), b)
    end
    codom2 = one(S1) ⊠ codomain(t2)
    dom2 = one(S1) ⊠ domain(t2)
    t2′ = similar(t2, codom2 ← dom2)
    for (c, b) in blocks(t2)
        copy!(block(t2′, unit(I1) ⊠ c), b)
    end
    return t1′ ⊗ t2′
end
