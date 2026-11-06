# Device-side tree transformers
# -----------------------------
# The positions and recoupling coefficients of a `TreeTransformer` are resolved into
# sizes/strides/offsets and packed into flat device vectors.

const TreeStructure{N} = Tuple{NTuple{N, Int}, Int}

# strides of a dense array of shape `sz`
_dense_strides(size::Dims) = Base.front((1, cumprod(size)...))

function _work_offsets(work)
    offsets = cumsum(work)
    pushfirst!(offsets, 0)
    total = pop!(offsets)
    return offsets, total
end

# finds the overall linear index in the output and input arrays corresponding to the **sublock**
# coordinates currently being worked on
@inline function _linear_index(coords::NTuple{N, Integer}, strides::NTuple{N, Int}, offset) where {N}
    # an explicit loop, since `sum` of an empty tuple (e.g. a full trace) does not compile for GPUs
    i = offset + 1
    for n in 1:N
        i += coords[n] * strides[n]
    end
    return i
end

# uploads preserve the element type, and must outlive any GPUArrays allocation-cache scope
_uploader(A) = (proto = A(undef, 0); x -> copy!(similar(proto, eltype(x), size(x)), x))
