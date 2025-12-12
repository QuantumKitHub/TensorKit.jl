const CuTensorMap{T, S, N₁, N₂} = TensorMap{T, S, N₁, N₂, CuVector{T, CUDA.DeviceMemory}}
const CuTensor{T, S, N} = CuTensorMap{T, S, N, 0}

const AdjointCuTensorMap{T, S, N₁, N₂} = AdjointTensorMap{T, S, N₁, N₂, CuTensorMap{T, S, N₁, N₂}}

TensorKit._tensormap_storagetype(::Type{A}) where {T, A <: CuArray{T}} = CuVector{T, CUDA.DeviceMemory}

function CuTensorMap{T, S, N₁, N₂}(t::TensorMap{T, S, N₁, N₂, A}) where {T, S, N₁, N₂, A}
    return CuTensorMap{T, S, N₁, N₂}(CuArray(t.data), t.space)
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
    end
end

for randfun in (:rand, :randn, :randisometry)
    randfun! = Symbol(randfun, :!)
    @eval begin
        # converting `codomain` and `domain` into `HomSpace`
        function $randfun(
                ::Type{A}, codomain::TensorSpace{S},
                domain::TensorSpace{S}
            ) where {A <: CuArray, S <: IndexSpace}
            return $randfun(A, codomain ← domain)
        end
        function $randfun(
                ::Type{T}, ::Type{A}, codomain::TensorSpace{S},
                domain::TensorSpace{S}
            ) where {T, S <: IndexSpace, A <: CuArray{T}}
            return $randfun(T, A, codomain ← domain)
        end
        function $randfun(
                rng::Random.AbstractRNG, ::Type{T}, ::Type{A},
                codomain::TensorSpace{S},
                domain::TensorSpace{S}
            ) where {T, S <: IndexSpace, A <: CuArray{T}}
            return $randfun(rng, T, A, codomain ← domain)
        end

        # accepting single `TensorSpace`
        $randfun(::Type{A}, codomain::TensorSpace) where {A <: CuArray} = $randfun(A, codomain ← one(codomain))
        function $randfun(::Type{T}, ::Type{A}, codomain::TensorSpace) where {T, A <: CuArray{T}}
            return $randfun(T, A, codomain ← one(codomain))
        end
        function $randfun(
                rng::Random.AbstractRNG, ::Type{T},
                ::Type{A}, codomain::TensorSpace
            ) where {T, A <: CuArray{T}}
            return $randfun(rng, T, A, codomain ← one(domain))
        end

        # filling in default eltype
        $randfun(::Type{A}, V::TensorMapSpace) where {A <: CuArray} = $randfun(eltype(A), A, V)
        function $randfun(rng::Random.AbstractRNG, ::Type{A}, V::TensorMapSpace) where {A <: CuArray}
            return $randfun(rng, eltype(A), A, V)
        end

        # filling in default rng
        function $randfun(::Type{T}, ::Type{A}, V::TensorMapSpace) where {T, A <: CuArray{T}}
            return $randfun(Random.default_rng(), T, A, V)
        end

        # implementation
        function $randfun(
                rng::Random.AbstractRNG, ::Type{T},
                ::Type{A}, V::TensorMapSpace
            ) where {T, A <: CuArray{T}}
            t = CuTensorMap{T}(undef, V)
            $randfun!(rng, t)
            return t
        end
    end
end

function Base.convert(::Type{CuTensorMap}, t::AbstractTensorMap)
    return copy!(CuTensorMap{scalartype(t)}(undef, space(t)), t)
end

# Scalar implementation
#-----------------------
function TensorKit.scalar(t::CuTensorMap)
    # TODO: should scalar only work if N₁ == N₂ == 0?
    return @allowscalar dim(codomain(t)) == dim(domain(t)) == 1 ?
        first(blocks(t))[2][1, 1] : throw(DimensionMismatch())
end

TensorKit.scalartype(A::StridedCuArray{T}) where {T} = T
TensorKit.scalartype(::Type{<:CuTensorMap{T}}) where {T} = T
TensorKit.scalartype(::Type{<:CuArray{T}}) where {T} = T

function TensorKit.similarstoragetype(TT::Type{<:CuTensorMap{TTT, S, N₁, N₂}}, ::Type{T}) where {TTT, T, S, N₁, N₂}
    return CuVector{T, CUDA.DeviceMemory}
end

function Base.convert(
        TT::Type{CuTensorMap{T, S, N₁, N₂}},
        t::AbstractTensorMap{<:Any, S, N₁, N₂}
    ) where {T, S, N₁, N₂}
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
        isherm = TensorKit.MatrixAlgebraKit.ishermitian(b; atol = eps(real(eltype(b))), rtol = eps(real(eltype(b))))
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

# CuTensorMap exponentation:
function TensorKit.exp!(t::CuTensorMap)
    domain(t) == codomain(t) ||
        error("Exponential of a tensor only exist when domain == codomain.")
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
            throw(SpaceMismatch("`$($sf)` of a tensor only exist when domain == codomain"))
        T = complex(float(scalartype(t)))
        tf = similar(t, T)
        for (c, b) in blocks(t)
            copy!(block(tf, c), parent($f(Hermitian(b))))
        end
        return tf
    end
end
