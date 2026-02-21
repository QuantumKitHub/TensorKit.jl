const CuTensorMap{T, S, N₁, N₂} = TensorMap{T, S, N₁, N₂, CuVector{T, CUDA.DeviceMemory}}
const CuTensor{T, S, N} = CuTensorMap{T, S, N, 0}

const AdjointCuTensorMap{T, S, N₁, N₂} = AdjointTensorMap{T, S, N₁, N₂, CuTensorMap{T, S, N₁, N₂}}

function CuTensorMap(t::TensorMap{T, S, N₁, N₂, A}) where {T, S, N₁, N₂, A}
    return CuTensorMap{T, S, N₁, N₂}(CuArray{T}(t.data), space(t))
end

# project_symmetric! doesn't yet work for GPU types, so do this on the host, then copy
function TensorKit.project_symmetric_and_check(::Type{T}, ::Type{A}, data::AbstractArray, V::TensorMapSpace; tol = sqrt(eps(real(float(eltype(data)))))) where {T, A <: CuVector{T}}
    h_t = TensorKit.TensorMapWithStorage{T, Vector{T}}(undef, V)
    h_t = TensorKit.project_symmetric!(h_t, Array(data))
    # verify result
    isapprox(Array(reshape(data, dims(h_t))), convert(Array, h_t); atol = tol) ||
        throw(ArgumentError("Data has non-zero elements at incompatible positions"))
    return TensorKit.TensorMapWithStorage{T, A}(A(h_t.data), V)
end

function TensorKit.blocktype(::Type{<:CuTensorMap{T, S}}) where {T, S}
    return CuMatrix{T, CUDA.DeviceMemory}
end

for (fname, felt) in ((:zeros, :zero), (:ones, :one))
    @eval begin
        function CUDA.$fname(
                codomain::TensorSpace{S},
                domain::TensorSpace{S} = one(codomain)
            ) where {S <: IndexSpace}
            return CUDA.$fname(codomain ← domain)
        end
        function CUDA.$fname(
                ::Type{T}, codomain::TensorSpace{S},
                domain::TensorSpace{S} = one(codomain)
            ) where {T, S <: IndexSpace}
            return CUDA.$fname(T, codomain ← domain)
        end
        CUDA.$fname(V::TensorMapSpace) = CUDA.$fname(Float64, V)
        function CUDA.$fname(::Type{T}, V::TensorMapSpace) where {T}
            t = CuTensorMap{T}(undef, V)
            fill!(t, $felt(T))
            return t
        end
    end
end

for randfun in (:curand, :curandn)
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
            t = CuTensorMap{T}(undef, V)
            $randfun!(rng, t)
            return t
        end

        function $randfun!(rng::Random.AbstractRNG, t::CuTensorMap)
            for (_, b) in blocks(t)
                $randfun!(rng, b)
            end
            return t
        end
    end
end

# Scalar implementation
#-----------------------
function TensorKit.scalar(t::CuTensorMap{T, S, 0, 0}) where {T, S}
    inds = findall(!iszero, t.data)
    return isempty(inds) ? zero(scalartype(t)) : @allowscalar @inbounds t.data[only(inds)]
end

function Base.convert(
        TT::Type{TensorMap{T, S, N₁, N₂, A}},
        t::TensorMap{T, S, N₁, N₂, AA}
    ) where {T, S, N₁, N₂, A <: CuArray{T}, AA}
    if typeof(t) === TT
        return t
    else
        tnew = TT(undef, space(t))
        return copy!(tnew, t)
    end
end

function Base.convert(
        TT::Type{TensorMap{T, S, N₁, N₂, A}},
        t::AdjointTensorMap
    ) where {T, S, N₁, N₂, A <: CuArray{T}}
    if typeof(t) === TT
        return t
    else
        tnew = TT(undef, space(t))
        return copy!(tnew, t)
    end
end

function LinearAlgebra.isposdef(t::CuTensorMap)
    domain(t) == codomain(t) ||
        throw(SpaceMismatch("`isposdef` requires domain and codomain to be the same"))
    InnerProductStyle(spacetype(t)) === EuclideanInnerProduct() || return false
    for (c, b) in blocks(t)
        # do our own hermitian check
        isherm = MatrixAlgebraKit.ishermitian(b)
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
        TT₁ <: CuTensorMap{TTT₁, S, N₁, N₂},
        TT₂ <: CuTensorMap{TTT₂, S, N₁, N₂},
    }
    T = TensorKit.VectorInterface.promote_add(TTT₁, TTT₂)
    return CuTensorMap{T, S, N₁, N₂}
end

TensorKit.promote_storage_rule(::Type{CuArray{T, N}}, ::Type{<:CuArray{T, N}}) where {T, N} =
    CuArray{T, N, CUDA.default_memory}
TensorKit.promote_storage_rule(::Type{<:CuArray{T, N}}, ::Type{CuArray{T, N}}) where {T, N} =
    CuArray{T, N, CUDA.default_memory}


# CuTensorMap exponentation:
function TensorKit.exp!(t::CuTensorMap)
    domain(t) == codomain(t) ||
        error("Exponential of a tensor only exist when domain == codomain.")
    !MatrixAlgebraKit.ishermitian(t) && throw(ArgumentError("`exp!` is currently only supported on hermitian CUDA tensors"))
    for (c, b) in blocks(t)
        copy!(b, parent(Base.exp(Hermitian(b))))
    end
    return t
end

# functions that don't map ℝ to (a subset of) ℝ
for f in (:sqrt, :log, :asin, :acos, :acosh, :atanh, :acoth)
    sf = string(f)
    @eval function Base.$f(t::CuTensorMap)
        domain(t) == codomain(t) ||
            throw(SpaceMismatch("`$($sf)` of a tensor only exists when domain == codomain"))
        !MatrixAlgebraKit.ishermitian(t) && throw(ArgumentError("`$($sf)` is currently only supported on hermitian CUDA tensors"))
        T = complex(float(scalartype(t)))
        tf = similar(t, T)
        for (c, b) in blocks(t)
            copy!(block(tf, c), parent($f(Hermitian(b))))
        end
        return tf
    end
end

function TensorKit._add_general_kernel_nonthreaded!(
        tdst::CuTensorMap, tsrc::CuTensorMap, p, transformer::TensorKit.GenericTreeTransformer, α, β, backend...
    )
    # preallocate buffers
    buffers = TensorKit.allocate_buffers(tdst, tsrc, transformer)

    for subtransformer in transformer.data
        # Special case without intermediate buffers whenever there is only a single block
        if length(subtransformer[1]) == 1
            TensorKit._add_transform_single!(tdst, tsrc, p, subtransformer, α, β, backend...)
        else
            cu_subtransformer = tuple(CUDA.adapt(CuArray, subtransformer[1]), subtransformer[2:end]...)
            TensorKit._add_transform_multi!(tdst, tsrc, p, cu_subtransformer, buffers, α, β, backend...)
        end
    end
    return nothing
end
