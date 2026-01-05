const ROCTensorMap{T, S, N₁, N₂} = TensorMap{T, S, N₁, N₂, ROCVector{T, AMDGPU.Mem.HIPBuffer}}
const ROCTensor{T, S, N} = ROCTensorMap{T, S, N, 0}

const AdjointROCTensorMap{T, S, N₁, N₂} = AdjointTensorMap{T, S, N₁, N₂, ROCTensorMap{T, S, N₁, N₂}}

function ROCTensorMap(t::TensorMap{T, S, N₁, N₂, A}) where {T, S, N₁, N₂, A}
    return ROCTensorMap{T, S, N₁, N₂}(ROCArray{T}(t.data), space(t))
end

# project_symmetric! doesn't yet work for GPU types, so do this on the host, then copy
function TensorKit.project_symmetric_and_check(::Type{T}, ::Type{A}, data::AbstractArray, V::TensorMapSpace; tol = sqrt(eps(real(float(eltype(data)))))) where {T, A <: ROCVector{T}}
    h_t = TensorKit.TensorMapWithStorage{T, Vector{T}}(undef, V)
    h_t = TensorKit.project_symmetric!(h_t, Array(data))
    # verify result
    isapprox(Array(reshape(data, dims(h_t))), convert(Array, h_t); atol = tol) ||
        throw(ArgumentError("Data has non-zero elements at incompatible positions"))
    return TensorKit.TensorMapWithStorage{T, A}(A(h_t.data), V)
end

for (fname, felt) in ((:zeros, :zero), (:ones, :one))
    @eval begin
        function AMDGPU.$fname(
                codomain::TensorSpace{S},
                domain::TensorSpace{S} = one(codomain)
            ) where {S <: IndexSpace}
            return AMDGPU.$fname(codomain ← domain)
        end
        function AMDGPU.$fname(
                ::Type{T}, codomain::TensorSpace{S},
                domain::TensorSpace{S} = one(codomain)
            ) where {T, S <: IndexSpace}
            return AMDGPU.$fname(T, codomain ← domain)
        end
        AMDGPU.$fname(V::TensorMapSpace) = AMDGPU.$fname(Float64, V)
        function AMDGPU.$fname(::Type{T}, V::TensorMapSpace) where {T}
            t = ROCTensorMap{T}(undef, V)
            fill!(t, $felt(T))
            return t
        end
    end
end

for randfun in (:rocrand, :rocrandn)
    randfun! = Symbol(randfun, :!)
    @eval begin
        # converting `codomain` and `domain` into `HomSpace`
        function $randfun(
                codomain::TensorSpace{S},
                domain::TensorSpace{S} = one(codomain),
            ) where {S <: IndexSpace}
            return $randfun(codomain ← domain)
        end
        function $randfun(
                ::Type{T}, codomain::TensorSpace{S},
                domain::TensorSpace{S} = one(codomain),
            ) where {T, S <: IndexSpace}
            return $randfun(T, codomain ← domain)
        end
        function $randfun(
                rng::Random.AbstractRNG, ::Type{T},
                codomain::TensorSpace{S},
                domain::TensorSpace{S} = one(codomain),
            ) where {T, S <: IndexSpace}
            return $randfun(rng, T, codomain ← domain)
        end

        # filling in default eltype
        $randfun(V::TensorMapSpace) = $randfun(Float64, V)
        function $randfun(rng::Random.AbstractRNG, V::TensorMapSpace)
            return $randfun(rng, Float64, V)
        end

        # filling in default rng
        function $randfun(::Type{T}, V::TensorMapSpace) where {T}
            return $randfun(Random.default_rng(), T, V)
        end

        # implementation
        function $randfun(
                rng::Random.AbstractRNG, ::Type{T},
                V::TensorMapSpace
            ) where {T}
            t = ROCTensorMap{T}(undef, V)
            $randfun!(rng, t)
            return t
        end

        function $randfun!(rng::Random.AbstractRNG, t::ROCTensorMap)
            for (_, b) in blocks(t)
                $randfun!(rng, b)
            end
            return t
        end
    end
end

# Scalar implementation
#-----------------------
function TensorKit.scalar(t::ROCTensorMap{T, S, 0, 0}) where {T, S}
    inds = findall(!iszero, t.data)
    return isempty(inds) ? zero(scalartype(t)) : @allowscalar @inbounds t.data[only(inds)]
end

function Base.convert(
        TT::Type{ROCTensorMap{T, S, N₁, N₂}},
        t::AbstractTensorMap{<:Any, S, N₁, N₂}
    ) where {T, S, N₁, N₂}
    if typeof(t) === TT
        return t
    else
        tnew = TT(undef, space(t))
        return copy!(tnew, t)
    end
end

function LinearAlgebra.isposdef(t::ROCTensorMap)
    domain(t) == codomain(t) ||
        throw(SpaceMismatch("`isposdef` requires domain and codomain to be the same"))
    InnerProductStyle(spacetype(t)) === EuclideanInnerProduct() || return false
    for (c, b) in blocks(t)
        # do our own hermitian check
        isherm = MatrixAlgebraKit.ishermitian(b; atol = eps(real(eltype(b))), rtol = eps(real(eltype(b))))
        isherm || return false
        isposdef(Hermitian(b)) || return false
    end
    return true
end

function Base.promote_rule(
        ::Type{<:TT₁},
        ::Type{<:TT₂}
    ) where {
        S, N₁, N₂, TTT₁, TTT₂,
        TT₁ <: ROCTensorMap{TTT₁, S, N₁, N₂},
        TT₂ <: ROCTensorMap{TTT₂, S, N₁, N₂},
    }
    T = TensorKit.VectorInterface.promote_add(TTT₁, TTT₂)
    return ROCTensorMap{T, S, N₁, N₂}
end

# ROCTensorMap exponentation:
function TensorKit.exp!(t::ROCTensorMap)
    domain(t) == codomain(t) ||
        error("Exponential of a tensor only exist when domain == codomain.")
    !MatrixAlgebraKit.ishermitian(t) && throw(ArgumentError("`exp!` is currently only supported on hermitian AMDGPU tensors"))
    for (c, b) in blocks(t)
        copy!(b, parent(Base.exp(Hermitian(b))))
    end
    return t
end

# functions that don't map ℝ to (a subset of) ℝ
for f in (:sqrt, :log, :asin, :acos, :acosh, :atanh, :acoth)
    sf = string(f)
    @eval function Base.$f(t::ROCTensorMap)
        domain(t) == codomain(t) ||
            throw(SpaceMismatch("`$($sf)` of a tensor only exists when domain == codomain"))
        !MatrixAlgebraKit.ishermitian(t) && throw(ArgumentError("`$($sf)` is currently only supported on hermitian AMDGPU tensors"))
        T = complex(float(scalartype(t)))
        tf = similar(t, T)
        for (c, b) in blocks(t)
            copy!(block(tf, c), parent($f(Hermitian(b))))
        end
        return tf
    end
end
