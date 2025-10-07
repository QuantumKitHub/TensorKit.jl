"""
    randisometry([::Type{T}=Float64], [::Type{A}=Matrix{T}], dims::Dims{2}) -> A
    randhaar([::Type{T}=Float64], [::Type{A}=Matrix{T}], dims::Dims{2}) -> A

Create a random isometry of size `dims`, uniformly distributed according to the Haar measure.

See also [`randuniform`](@ref) and [`randnormal`](@ref).
"""
randisometry(dims::Base.Dims{2}) = randisometry(Float64, Matrix{Float64}, dims)
function randisometry(::Type{T}, dims::Base.Dims{2}) where {T <: Number}
    return randisometry(Random.default_rng(), T, dims)
end
function randisometry(::Type{T}, ::Type{A}, dims::Base.Dims{2}) where {T <: Number, A <: AbstractArray{T}}
    return randisometry(Random.default_rng(), T, A, dims)
end
function randisometry(rng::Random.AbstractRNG, ::Type{T}, ::Type{A}, dims::Base.Dims{2}) where {T <: Number, A <: AbstractArray{T}}
    return randisometry!(rng, A(undef, dims))
end

randisometry!(A::AbstractMatrix) = randisometry!(Random.default_rng(), A)
function randisometry!(rng::Random.AbstractRNG, A::AbstractMatrix)
    dims = size(A)
    dims[1] >= dims[2] ||
        throw(DimensionMismatch("cannot create isometric matrix with dimensions $dims; isometry needs to be tall or square"))
    Q, = qr_compact!(Random.randn!(rng, A); positive = true)
    return copy!(A, Q)
end
