module TensorKitJLD2Ext

using TensorKit
using TensorKit: AdjointTensorMap
import TensorKit: save_tensor, load_tensor
import JLD2

# TensorMap IO
#=============#

const TENSORMAP_FILE_FORMAT = "TensorKit.AbstractTensorMap"
const TENSORMAP_FILE_VERSION = UInt16(1)

const _FUSIONTREE_TABLE_FIELDS = (:uncoupled, :coupled, :isdual, :innerlines, :vertices)

"""Return the position of a fusion tree, adding it to the table when necessary."""
function _intern_fusiontree!(trees::AbstractVector, tree::FusionTree)
    index = findfirst(==(tree), trees)
    if isnothing(index)
        push!(trees, tree)
        return length(trees)
    end
    return index
end

"""Encode fusion trees as columnar arrays of their semantic fields."""
function _encode_fusiontrees(trees::AbstractVector, ::Type{I}, numlegs::Int) where {I <: Sector}
    numtrees = length(trees)
    numinner = max(0, numlegs - 2)
    numvertices = max(0, numlegs - 1)
    uncoupled = Matrix{I}(undef, numlegs, numtrees)
    coupled = Vector{I}(undef, numtrees)
    isdual = falses(numlegs, numtrees)
    innerlines = Matrix{I}(undef, numinner, numtrees)
    vertices = Matrix{Int}(undef, numvertices, numtrees)
    for (column, tree) in enumerate(trees)
        length(tree.uncoupled) == numlegs ||
            error("inconsistent fusion-tree leg count while saving")
        length(tree.innerlines) == numinner ||
            error("inconsistent fusion-tree inner-line count while saving")
        length(tree.vertices) == numvertices ||
            error("inconsistent fusion-tree vertex count while saving")
        uncoupled[:, column] .= tree.uncoupled
        coupled[column] = tree.coupled
        isdual[:, column] .= tree.isdual
        innerlines[:, column] .= tree.innerlines
        vertices[:, column] .= tree.vertices
    end
    return (; uncoupled, coupled, isdual, innerlines, vertices)
end

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

"""Decode a columnar fusion-tree table and validate its basic representation."""
function _decode_fusiontrees(table, ::Type{I}, numlegs::Int, description::AbstractString) where {I <: Sector}
    _require_record_fields(table, _FUSIONTREE_TABLE_FIELDS, description)
    table.uncoupled isa Matrix{I} ||
        throw(ArgumentError("serialized $description has invalid uncoupled sectors"))
    table.coupled isa Vector{I} ||
        throw(ArgumentError("serialized $description has invalid coupled sectors"))
    table.isdual isa BitMatrix ||
        throw(ArgumentError("serialized $description has invalid duality flags"))
    table.innerlines isa Matrix{I} ||
        throw(ArgumentError("serialized $description has invalid inner lines"))
    table.vertices isa Matrix{Int} ||
        throw(ArgumentError("serialized $description has invalid vertices"))

    numtrees = length(table.coupled)
    expected_sizes = (
        (numlegs, numtrees),
        (numlegs, numtrees),
        (max(0, numlegs - 2), numtrees),
        (max(0, numlegs - 1), numtrees),
    )
    actual_sizes = (
        size(table.uncoupled), size(table.isdual),
        size(table.innerlines), size(table.vertices),
    )
    actual_sizes == expected_sizes ||
        throw(DimensionMismatch("serialized $description has inconsistent table dimensions"))

    trees = Vector{FusionTree{I, numlegs}}(undef, numtrees)
    for column in 1:numtrees
        uncoupled = ntuple(row -> table.uncoupled[row, column], numlegs)
        isdual = ntuple(row -> table.isdual[row, column], numlegs)
        innerlines = ntuple(row -> table.innerlines[row, column], max(0, numlegs - 2))
        vertices = ntuple(row -> table.vertices[row, column], max(0, numlegs - 1))
        trees[column] = try
            FusionTree{I}(uncoupled, table.coupled[column], isdual, innerlines, vertices)
        catch error
            message = sprint(showerror, error)
            throw(ArgumentError("serialized $description contains an invalid fusion tree: $message"))
        end
    end
    _check_unique_values(trees, description)
    return trees
end

"""Decode explicit fusion-tree pair identifiers and validate them against a tensor-map space."""
function _decode_fusiontree_pairs(pair_ids, codomain_trees, domain_trees, tensor_space::TensorMapSpace)
    pair_ids isa Matrix{Int} ||
        throw(ArgumentError("serialized tensor has invalid fusion-tree pair identifiers"))
    size(pair_ids, 1) == 2 ||
        throw(DimensionMismatch("serialized fusion-tree pair identifiers must have two rows"))

    numpairs = size(pair_ids, 2)
    pairs = Vector{Tuple{eltype(codomain_trees), eltype(domain_trees)}}(undef, numpairs)
    for column in 1:numpairs
        codomain_id = pair_ids[1, column]
        domain_id = pair_ids[2, column]
        checkbounds(Bool, codomain_trees, codomain_id) ||
            throw(ArgumentError("serialized tensor has an out-of-range codomain fusion-tree identifier"))
        checkbounds(Bool, domain_trees, domain_id) ||
            throw(ArgumentError("serialized tensor has an out-of-range domain fusion-tree identifier"))
        pairs[column] = (codomain_trees[codomain_id], domain_trees[domain_id])
    end
    _check_unique_values(pairs, "fusion-tree pairs")

    all(id -> id in @view(pair_ids[1, :]), eachindex(codomain_trees)) ||
        throw(ArgumentError("serialized tensor contains an unused codomain fusion tree"))
    all(id -> id in @view(pair_ids[2, :]), eachindex(domain_trees)) ||
        throw(ArgumentError("serialized tensor contains an unused domain fusion tree"))

    expected_pairs = collect(fusiontrees(tensor_space))
    length(pairs) == length(expected_pairs) &&
        all(pair -> any(==(pair), expected_pairs), pairs) ||
        throw(ArgumentError("serialized fusion-tree pairs do not match the tensor-map space"))
    return pairs
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
    data = Vector{T}(undef, length(t.data))
    offset = 0
    for (column, (codomain_tree, domain_tree)) in enumerate(tree_pairs)
        pair_ids[1, column] = _intern_fusiontree!(codomain_trees, codomain_tree)
        pair_ids[2, column] = _intern_fusiontree!(domain_trees, domain_tree)
        source = subblock(t, (codomain_tree, domain_tree))
        shapes[:, column] .= size(source)
        elements = vec(Array(source))
        offset + length(elements) <= length(data) ||
            error("inconsistent TensorMap subblock storage")
        copyto!(data, offset + 1, elements, 1, length(elements))
        offset += length(elements)
    end
    offset == length(data) || error("inconsistent TensorMap subblock storage")
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
    data = Vector{T}(undef, length(t.data))
    offset = 0
    for (column, (codomain_tree, domain_tree)) in enumerate(tree_pairs)
        pair_ids[1, column] = _intern_fusiontree!(trees, codomain_tree)
        pair_ids[2, column] = _intern_fusiontree!(trees, domain_tree)
        elements = Vector(subblock(t, (codomain_tree, domain_tree)).diag)
        lengths[column] = length(elements)
        offset + length(elements) <= length(data) ||
            error("inconsistent DiagonalTensorMap subblock storage")
        copyto!(data, offset + 1, elements, 1, length(elements))
        offset += length(elements)
    end
    offset == length(data) || error("inconsistent DiagonalTensorMap subblock storage")
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

function save_tensor(path::AbstractString, tensor::AbstractTensorMap)
    record = _pack_tensormap(tensor)
    destination = abspath(path)
    temporary, io = mktemp(dirname(destination))
    close(io)
    committed = false
    try
        JLD2.jldopen(temporary, "w") do file
            file["format"] = TENSORMAP_FILE_FORMAT
            file["version"] = TENSORMAP_FILE_VERSION
            file["tensor"] = record
        end
        mv(temporary, destination; force = true)
        committed = true
    finally
        !committed && isfile(temporary) && rm(temporary)
    end
    return nothing
end

function load_tensor(path::AbstractString)
    record = JLD2.jldopen(path, "r") do file
        all(key -> haskey(file, key), ("format", "version", "tensor")) ||
            throw(ArgumentError("file is not a TensorKit tensor-map file"))
        file["format"] == TENSORMAP_FILE_FORMAT ||
            throw(ArgumentError("file has an invalid TensorKit tensor-map format marker"))
        version = file["version"]
        version == TENSORMAP_FILE_VERSION ||
            throw(ArgumentError("unsupported TensorKit tensor-map file version $version"))
        return file["tensor"]
    end
    return _unpack_tensormap(record)
end

end
