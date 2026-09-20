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
    used_codomain = falses(length(codomain_trees))
    used_domain = falses(length(domain_trees))
    for column in 1:numpairs
        codomain_id = pair_ids[1, column]
        domain_id = pair_ids[2, column]
        checkbounds(Bool, codomain_trees, codomain_id) ||
            throw(ArgumentError("serialized tensor has an out-of-range codomain fusion-tree identifier"))
        checkbounds(Bool, domain_trees, domain_id) ||
            throw(ArgumentError("serialized tensor has an out-of-range domain fusion-tree identifier"))
        pairs[column] = (codomain_trees[codomain_id], domain_trees[domain_id])
        used_codomain[codomain_id] = true
        used_domain[domain_id] = true
    end
    _check_unique_values(pairs, "fusion-tree pairs")

    all(used_codomain) ||
        throw(ArgumentError("serialized tensor contains an unused codomain fusion tree"))
    all(used_domain) ||
        throw(ArgumentError("serialized tensor contains an unused domain fusion tree"))

    expected_pairs = collect(fusiontrees(tensor_space))
    length(pairs) == length(expected_pairs) &&
        all(pair -> any(==(pair), expected_pairs), pairs) ||
        throw(ArgumentError("serialized fusion-tree pairs do not match the tensor-map space"))
    return pairs
end
