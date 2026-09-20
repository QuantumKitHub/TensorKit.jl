"""Require a named tuple to contain all fields used by a serialized record."""
function _require_record_fields(record, fields::Tuple, description::AbstractString)
    record isa NamedTuple ||
        throw(ArgumentError("serialized $description must be a NamedTuple"))
    missing = filter(field -> !hasproperty(record, field), fields)
    isempty(missing) ||
        throw(ArgumentError("serialized $description is missing fields $(join(missing, ", "))"))
    return nothing
end

"""Check that values are unique using equality without imposing an ordering or hash contract."""
function _check_unique_values(values, description::AbstractString)
    for index in eachindex(values)
        any(previous -> previous == values[index], @view(values[firstindex(values):(index - 1)])) &&
            throw(ArgumentError("serialized tensor contains duplicate $description"))
    end
    return nothing
end

"""Pack a dense tensor map using explicit fusion-tree pairs and CPU subblock elements."""
function _pack_tensormap(t::TensorMap{T}) where {T}
    I = sectortype(t)
    Nout = numout(t)
    Nin = numin(t)
    tree_pairs = fusiontrees(t)
    numpairs = length(tree_pairs)
    codomain_trees = FusionTree{I, Nout}[]
    domain_trees = FusionTree{I, Nin}[]
    pair_ids = Matrix{Int}(undef, 2, numpairs)
    shapes = Matrix{Int}(undef, numind(t), numpairs)
    data = sizehint!(T[], length(t.data))
    for (column, (codomain_tree, domain_tree)) in enumerate(tree_pairs)
        pair_ids[1, column] = _intern_fusiontree!(codomain_trees, codomain_tree)
        pair_ids[2, column] = _intern_fusiontree!(domain_trees, domain_tree)
        source = subblock(t, (codomain_tree, domain_tree))
        shapes[:, column] .= size(source)
        append!(data, vec(Array(source)))
    end
    length(data) == length(t.data) || error("inconsistent TensorMap subblock storage")
    codomain_table = _encode_fusiontrees(codomain_trees, I, Nout)
    domain_table = _encode_fusiontrees(domain_trees, I, Nin)
    return (;
        kind = :dense, space = space(t), codomain_trees = codomain_table,
        domain_trees = domain_table, pair_ids, shapes, data,
    )
end

"""Pack a diagonal tensor map using fusion-tree labels and compact diagonal elements."""
function _pack_tensormap(t::DiagonalTensorMap{T}) where {T}
    I = sectortype(t)
    tree_pairs = fusiontrees(t)
    numpairs = length(tree_pairs)
    trees = FusionTree{I, 1}[]
    pair_ids = Matrix{Int}(undef, 2, numpairs)
    lengths = Vector{Int}(undef, numpairs)
    data = sizehint!(T[], length(t.data))
    for (column, (codomain_tree, domain_tree)) in enumerate(tree_pairs)
        pair_ids[1, column] = _intern_fusiontree!(trees, codomain_tree)
        pair_ids[2, column] = _intern_fusiontree!(trees, domain_tree)
        elements = Vector(subblock(t, (codomain_tree, domain_tree)).diag)
        lengths[column] = length(elements)
        append!(data, elements)
    end
    length(data) == length(t.data) || error("inconsistent DiagonalTensorMap subblock storage")
    tree_table = _encode_fusiontrees(trees, I, 1)
    return (; kind = :diagonal, domain = only(domain(t)), trees = tree_table, pair_ids, lengths, data)
end

"""Pack a braiding tensor using only the spaces and orientation that define it."""
function _pack_tensormap(t::BraidingTensor{T}) where {T}
    return (; kind = :braiding, V1 = t.V1, V2 = t.V2, adjoint = t.adjoint, scalartype = T)
end

"""Reject lazy adjoints so that saving never hides an implicit materialization choice."""
function _pack_tensormap(::AdjointTensorMap)
    throw(ArgumentError("AdjointTensorMap must be materialized with `convert(TensorMap, tensor)` before saving"))
end

"""Reject tensor-map implementations without an explicit stable serialization record."""
function _pack_tensormap(t::AbstractTensorMap)
    throw(ArgumentError("saving $(typeof(t)) is not supported; materialize it as a built-in TensorMap type first"))
end

"""Reconstruct a dense tensor map from an order-independent version-one record."""
function _unpack_dense_tensormap(record)
    fields = (:space, :codomain_trees, :domain_trees, :pair_ids, :shapes, :data)
    _require_record_fields(record, fields, "dense TensorMap record")
    record.space isa TensorMapSpace ||
        throw(ArgumentError("serialized TensorMap has an invalid tensor-map space"))
    record.data isa Vector ||
        throw(ArgumentError("serialized TensorMap data must be a Vector"))
    eltype(record.data) <: Number ||
        throw(ArgumentError("serialized TensorMap has an invalid scalar type"))

    I = sectortype(record.space)
    codomain_trees = _decode_fusiontrees(
        record.codomain_trees, I, numout(record.space), "codomain fusion trees"
    )
    domain_trees = _decode_fusiontrees(
        record.domain_trees, I, numin(record.space), "domain fusion trees"
    )
    pairs = _decode_fusiontree_pairs(
        record.pair_ids, codomain_trees, domain_trees, record.space
    )
    record.shapes isa Matrix{Int} ||
        throw(ArgumentError("serialized TensorMap has invalid subblock shapes"))
    size(record.shapes) == (numind(record.space), length(pairs)) ||
        throw(DimensionMismatch("serialized TensorMap has inconsistent subblock shape metadata"))

    T = eltype(record.data)
    tensor = TensorMap{T}(undef, record.space)
    offset = 0
    for (column, pair) in enumerate(pairs)
        shape = Tuple(@view record.shapes[:, column])
        all(>=(0), shape) ||
            throw(ArgumentError("serialized TensorMap contains a negative subblock dimension"))
        destination = subblock(tensor, pair)
        size(destination) == shape ||
            throw(DimensionMismatch("serialized TensorMap subblock has shape $shape, expected $(size(destination))"))
        blocklength = length(destination)
        blocklength <= length(record.data) - offset ||
            throw(DimensionMismatch("serialized TensorMap data is shorter than its subblock metadata"))
        source = reshape(@view(record.data[(offset + 1):(offset + blocklength)]), shape)
        copyto!(destination, source)
        offset += blocklength
    end
    offset == length(record.data) ||
        throw(DimensionMismatch("serialized TensorMap data is longer than its subblock metadata"))
    return tensor
end

"""Reconstruct a compact diagonal tensor map from an order-independent version-one record."""
function _unpack_diagonal_tensormap(record)
    fields = (:domain, :trees, :pair_ids, :lengths, :data)
    _require_record_fields(record, fields, "DiagonalTensorMap record")
    record.domain isa IndexSpace ||
        throw(ArgumentError("serialized DiagonalTensorMap has an invalid domain"))
    record.data isa Vector ||
        throw(ArgumentError("serialized DiagonalTensorMap data must be a Vector"))
    eltype(record.data) <: Number ||
        throw(ArgumentError("serialized DiagonalTensorMap has an invalid scalar type"))
    record.lengths isa Vector{Int} ||
        throw(ArgumentError("serialized DiagonalTensorMap has invalid segment lengths"))

    I = sectortype(record.domain)
    trees = _decode_fusiontrees(record.trees, I, 1, "diagonal fusion trees")
    tensor_space = record.domain ← record.domain
    pairs = _decode_fusiontree_pairs(record.pair_ids, trees, trees, tensor_space)
    length(record.lengths) == length(pairs) ||
        throw(ArgumentError("serialized DiagonalTensorMap has inconsistent segment metadata"))

    T = eltype(record.data)
    tensor = DiagonalTensorMap{T}(undef, record.domain)
    offset = 0
    for (column, pair) in enumerate(pairs)
        blocklength = record.lengths[column]
        blocklength >= 0 ||
            throw(ArgumentError("serialized DiagonalTensorMap contains a negative segment length"))
        destination = subblock(tensor, pair).diag
        length(destination) == blocklength ||
            throw(DimensionMismatch("serialized diagonal segment has length $blocklength, expected $(length(destination))"))
        blocklength <= length(record.data) - offset ||
            throw(DimensionMismatch("serialized DiagonalTensorMap data is shorter than its segment metadata"))
        copyto!(destination, @view(record.data[(offset + 1):(offset + blocklength)]))
        offset += blocklength
    end
    offset == length(record.data) ||
        throw(DimensionMismatch("serialized DiagonalTensorMap data is longer than its segment metadata"))
    return tensor
end

"""Reconstruct a braiding tensor from its structural version-one record."""
function _unpack_braiding_tensormap(record)
    fields = (:V1, :V2, :adjoint, :scalartype)
    _require_record_fields(record, fields, "BraidingTensor record")
    record.V1 isa IndexSpace && record.V2 isa IndexSpace ||
        throw(ArgumentError("serialized BraidingTensor has invalid spaces"))
    record.adjoint isa Bool ||
        throw(ArgumentError("serialized BraidingTensor has an invalid orientation flag"))
    record.scalartype isa Type && record.scalartype <: Number ||
        throw(ArgumentError("serialized BraidingTensor has an invalid scalar type"))
    return try
        BraidingTensor{record.scalartype}(record.V1, record.V2, record.adjoint)
    catch error
        message = sprint(showerror, error)
        throw(ArgumentError("serialized BraidingTensor is inconsistent: $message"))
    end
end

"""Dispatch a version-one basic Julia record to its tensor-map decoder."""
function _unpack_tensormap(record)
    _require_record_fields(record, (:kind,), "tensor record")
    record.kind === :dense && return _unpack_dense_tensormap(record)
    record.kind === :diagonal && return _unpack_diagonal_tensormap(record)
    record.kind === :braiding && return _unpack_braiding_tensormap(record)
    throw(ArgumentError("unsupported TensorKit tensor record kind $(repr(record.kind))"))
end
