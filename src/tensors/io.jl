# TensorMap IO
#=============#

const TENSORMAP_FILE_FORMAT = "TensorKit.AbstractTensorMap"
const TENSORMAP_FILE_VERSION = UInt16(1)

abstract type AbstractTensorMapRecordV1 end

struct DenseTensorMapRecordV1{S, I, T} <: AbstractTensorMapRecordV1
    space::S
    sectors::Vector{I}
    blockshapes::Vector{Tuple{Int, Int}}
    data::Vector{T}
end

struct DiagonalTensorMapRecordV1{S, I, T} <: AbstractTensorMapRecordV1
    domain::S
    sectors::Vector{I}
    blocklengths::Vector{Int}
    data::Vector{T}
end

struct BraidingTensorRecordV1{S, T} <: AbstractTensorMapRecordV1
    V1::S
    V2::S
    adjoint::Bool
    scalartype::Type{T}
end

"""Pack a dense tensor map into the portable version-one representation."""
function _pack_tensormap(t::TensorMap{T}) where {T}
    I = sectortype(t)
    sectors = I[]
    blockshapes = Tuple{Int, Int}[]
    data = Vector{T}(undef, dim(t))
    offset = 0
    for (c, b) in blocks(t)
        push!(sectors, c)
        push!(blockshapes, size(b))
        blockdata = vec(Array(b))
        copyto!(data, offset + 1, blockdata, 1, length(blockdata))
        offset += length(blockdata)
    end
    offset == length(data) || error("inconsistent TensorMap block storage")
    return DenseTensorMapRecordV1(space(t), sectors, blockshapes, data)
end

"""Pack a diagonal tensor map without expanding its zero off-diagonal entries."""
function _pack_tensormap(t::DiagonalTensorMap{T}) where {T}
    I = sectortype(t)
    sectors = I[]
    blocklengths = Int[]
    data = Vector{T}(undef, length(t.data))
    offset = 0
    for (c, b) in blocks(t)
        diagonal = Array(b.diag)
        push!(sectors, c)
        push!(blocklengths, length(diagonal))
        copyto!(data, offset + 1, diagonal, 1, length(diagonal))
        offset += length(diagonal)
    end
    offset == length(data) || error("inconsistent DiagonalTensorMap block storage")
    return DiagonalTensorMapRecordV1(only(domain(t)), sectors, blocklengths, data)
end

"""Pack a braiding tensor using only the spaces and orientation that define it."""
function _pack_tensormap(t::BraidingTensor{T}) where {T}
    return BraidingTensorRecordV1(t.V1, t.V2, t.adjoint, T)
end

"""Reject lazy adjoints so that saving never hides an implicit materialization choice."""
function _pack_tensormap(::AdjointTensorMap)
    throw(ArgumentError("AdjointTensorMap must be materialized with `convert(TensorMap, tensor)` before saving"))
end

"""Reject tensor-map implementations without an explicit stable serialization record."""
function _pack_tensormap(t::AbstractTensorMap)
    throw(ArgumentError("saving $(typeof(t)) is not supported; materialize it as a built-in TensorMap type first"))
end

"""Check that serialized block labels are unique."""
function _check_unique_sectors(sectors)
    length(unique(sectors)) == length(sectors) ||
        throw(ArgumentError("serialized tensor contains duplicate block sectors"))
    return nothing
end

"""Reconstruct a dense tensor map from a validated version-one record."""
function _unpack_tensormap(record::DenseTensorMapRecordV1{S, I, T}) where {S, I, T}
    length(record.sectors) == length(record.blockshapes) ||
        throw(ArgumentError("serialized TensorMap has inconsistent block metadata"))
    _check_unique_sectors(record.sectors)

    tensor = TensorMap{T}(undef, record.space)
    expected_sectors = collect(blocksectors(tensor))
    length(record.sectors) == length(expected_sectors) &&
        all(c -> c in expected_sectors, record.sectors) ||
        throw(ArgumentError("serialized TensorMap block sectors do not match its space"))

    offset = 0
    for (c, shape) in zip(record.sectors, record.blockshapes)
        destination = block(tensor, c)
        size(destination) == shape ||
            throw(DimensionMismatch("serialized TensorMap block for sector $c has shape $shape, expected $(size(destination))"))
        blocklength = prod(shape)
        offset + blocklength <= length(record.data) ||
            throw(DimensionMismatch("serialized TensorMap data is shorter than its block metadata"))
        copyto!(destination, reshape(view(record.data, (offset + 1):(offset + blocklength)), shape))
        offset += blocklength
    end
    offset == length(record.data) ||
        throw(DimensionMismatch("serialized TensorMap data is longer than its block metadata"))
    return tensor
end

"""Reconstruct a diagonal tensor map from a validated compact record."""
function _unpack_tensormap(record::DiagonalTensorMapRecordV1{S, I, T}) where {S, I, T}
    length(record.sectors) == length(record.blocklengths) ||
        throw(ArgumentError("serialized DiagonalTensorMap has inconsistent block metadata"))
    _check_unique_sectors(record.sectors)

    tensor = DiagonalTensorMap{T}(undef, record.domain)
    expected_sectors = collect(blocksectors(tensor))
    length(record.sectors) == length(expected_sectors) &&
        all(c -> c in expected_sectors, record.sectors) ||
        throw(ArgumentError("serialized DiagonalTensorMap block sectors do not match its space"))

    offset = 0
    for (c, blocklength) in zip(record.sectors, record.blocklengths)
        blocklength >= 0 || throw(ArgumentError("serialized diagonal block length is negative"))
        destination = block(tensor, c).diag
        length(destination) == blocklength ||
            throw(DimensionMismatch("serialized DiagonalTensorMap block for sector $c has length $blocklength, expected $(length(destination))"))
        offset + blocklength <= length(record.data) ||
            throw(DimensionMismatch("serialized DiagonalTensorMap data is shorter than its block metadata"))
        copyto!(destination, view(record.data, (offset + 1):(offset + blocklength)))
        offset += blocklength
    end
    offset == length(record.data) ||
        throw(DimensionMismatch("serialized DiagonalTensorMap data is longer than its block metadata"))
    return tensor
end

"""Reconstruct a braiding tensor from its structural version-one record."""
function _unpack_tensormap(record::BraidingTensorRecordV1{S, T}) where {S, T}
    record.scalartype === T || throw(ArgumentError("serialized BraidingTensor has an inconsistent scalar type"))
    return BraidingTensor{T}(record.V1, record.V2, record.adjoint)
end

"""Reject unknown serialization record types."""
function _unpack_tensormap(record)
    throw(ArgumentError("unsupported TensorKit tensor record $(typeof(record))"))
end

"""
    save(path::AbstractString, tensor::AbstractTensorMap)

Save one materialized tensor map to `path` using TensorKit's versioned JLD2 format.
Numerical data is copied to CPU storage, and an existing file is replaced.
"""
function save(path::AbstractString, tensor::AbstractTensorMap)
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

"""
    load(path::AbstractString) -> AbstractTensorMap

Load one tensor map saved with [`save`](@ref), using CPU storage for numerical data.
"""
function load(path::AbstractString)
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
