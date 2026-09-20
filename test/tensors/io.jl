using Test
import JLD2
using TensorKit

struct UnregisteredTensorMap <: AbstractTensorMap{Float64, ComplexSpace, 1, 0} end

struct TestDenseVector{T} <: DenseVector{T}
    data::Vector{T}
end
Base.size(vector::TestDenseVector) = size(vector.data)
Base.IndexStyle(::Type{<:TestDenseVector}) = IndexLinear()
Base.getindex(vector::TestDenseVector, index::Int) = vector.data[index]
Base.setindex!(vector::TestDenseVector, value, index::Int) = (vector.data[index] = value)

"""Write a raw TensorKit tensor record for malformed-file and permutation tests."""
function write_record(path, record; format = TensorKit.TENSORMAP_FILE_FORMAT, version = TensorKit.TENSORMAP_FILE_VERSION)
    return JLD2.jldopen(path, "w") do file
        file["format"] = format
        file["version"] = version
        file["tensor"] = record
    end
end

"""Reverse every order-bearing part of a tensor record while preserving its semantics."""
function reverse_record(record)
    permute_table = (table, permutation) -> (
        uncoupled = table.uncoupled[:, permutation],
        coupled = table.coupled[permutation],
        isdual = table.isdual[:, permutation],
        innerlines = table.innerlines[:, permutation],
        vertices = table.vertices[:, permutation],
    )
    pair_permutation = reverse(axes(record.pair_ids, 2))
    lengths = record.kind === :dense ?
        [prod(@view(record.shapes[:, column])) for column in axes(record.shapes, 2)] :
        record.lengths
    offsets = cumsum(vcat(0, lengths))
    data = similar(record.data)
    destination = 0
    for source_column in pair_permutation
        source_range = (offsets[source_column] + 1):offsets[source_column + 1]
        copyto!(data, destination + 1, record.data, first(source_range), length(source_range))
        destination += length(source_range)
    end

    pair_ids = copy(record.pair_ids)
    if record.kind === :dense
        codomain_permutation = reverse(eachindex(record.codomain_trees.coupled))
        domain_permutation = reverse(eachindex(record.domain_trees.coupled))
        pair_ids[1, :] .= invperm(codomain_permutation)[pair_ids[1, :]]
        pair_ids[2, :] .= invperm(domain_permutation)[pair_ids[2, :]]
        return merge(record, (
            codomain_trees = permute_table(record.codomain_trees, codomain_permutation),
            domain_trees = permute_table(record.domain_trees, domain_permutation),
            pair_ids = pair_ids[:, pair_permutation],
            shapes = record.shapes[:, pair_permutation],
            data,
        ))
    end
    tree_permutation = reverse(eachindex(record.trees.coupled))
    remapping = invperm(tree_permutation)
    pair_ids .= remapping[pair_ids]
    return merge(record, (
        trees = permute_table(record.trees, tree_permutation),
        pair_ids = pair_ids[:, pair_permutation],
        lengths = record.lengths[pair_permutation],
        data,
    ))
end

@testset "TensorMap save_tensor and load_tensor" begin
    @test :save_tensor in names(TensorKit)
    @test :load_tensor in names(TensorKit)
    @test :save ∉ names(TensorKit)
    @test :load ∉ names(TensorKit)

    spacelists = (
        TestSetup.Vtr,
        TestSetup.VRepℤ₂,
        TestSetup.VRepSU₂,
        TestSetup.VRepA4,
        TestSetup.VIBM,
    )
    mktempdir() do directory
        for (index, spaces) in enumerate(spacelists)
            V1, V2, V3, V4, V5 = spaces
            tensor = randn(ComplexF64, V1 ⊗ V2 ← (V3 ⊗ V4 ⊗ V5)')
            path = joinpath(directory, "tensor-$index.jld2")
            @test save_tensor(path, tensor) === nothing
            restored = load_tensor(path)
            @test restored isa TensorMap
            @test storagetype(restored) === Vector{ComplexF64}
            @test space(restored) == space(tensor)
            @test restored == tensor
        end

        tensor = randn(Float64, ℂ^2 ⊗ ℂ^3)
        path = joinpath(directory, "plain-tensor.jld2")
        save_tensor(path, tensor)
        restored = load_tensor(path)
        @test restored isa Tensor
        @test restored == tensor

        scalar_space = one(ℂ^1)
        scalar = randn(ComplexF64, scalar_space ← scalar_space)
        path = joinpath(directory, "scalar.jld2")
        save_tensor(path, scalar)
        restored_scalar = load_tensor(path)
        @test numout(restored_scalar) == numin(restored_scalar) == 0
        @test restored_scalar == scalar

        empty_tensor = randn(Float64, zero(ℂ^2) ← ℂ^2)
        path = joinpath(directory, "empty.jld2")
        save_tensor(path, empty_tensor)
        restored_empty = load_tensor(path)
        @test restored_empty == empty_tensor
        @test isempty(restored_empty.data)

        source = randn(Float64, ℂ^3 ← ℂ^2)
        custom_data = TestDenseVector(copy(source.data))
        custom = TensorMap{Float64, ComplexSpace, 1, 1, typeof(custom_data)}(custom_data, space(source))
        @test storagetype(custom) === TestDenseVector{Float64}
        path = joinpath(directory, "custom.jld2")
        save_tensor(path, custom)
        restored_custom = load_tensor(path)
        @test storagetype(restored_custom) === Vector{Float64}
        @test restored_custom == custom

        diagonal_space = Vect[SU2Irrep](0 => 3, 1 // 2 => 2, 1 => 1)'
        diagonal = DiagonalTensorMap(randn(ComplexF64, reduceddim(diagonal_space)), diagonal_space)
        for (index, value) in enumerate((diagonal, diagonal'))
            path = joinpath(directory, "diagonal-$index.jld2")
            save_tensor(path, value)
            restored_diagonal = load_tensor(path)
            @test restored_diagonal isa DiagonalTensorMap
            @test storagetype(restored_diagonal) === Vector{ComplexF64}
            @test restored_diagonal == value
        end

        braid_space = Vect[FibonacciAnyon](:I => 3, :τ => 2)
        for braiding in (BraidingTensor(braid_space, braid_space'), BraidingTensor(braid_space, braid_space')')
            path = joinpath(directory, "braiding-$(braiding.adjoint).jld2")
            save_tensor(path, braiding)
            restored_braiding = load_tensor(path)
            @test restored_braiding isa BraidingTensor
            @test storagetype(restored_braiding) === Vector{eltype(braiding)}
            @test restored_braiding.V1 == braiding.V1
            @test restored_braiding.V2 == braiding.V2
            @test restored_braiding.adjoint == braiding.adjoint
            @test TensorMap(restored_braiding) == TensorMap(braiding)
        end

        @test convert(TensorMap, convert(Dict, source)) == source
        @test_throws ArgumentError save_tensor(joinpath(directory, "adjoint.jld2"), source')
        @test_throws ArgumentError save_tensor(joinpath(directory, "unsupported.jld2"), UnregisteredTensorMap())
    end
end

@testset "TensorMap record representation" begin
    V1, V2, V3, V4, V5 = TestSetup.VRepA4
    tensor = randn(ComplexF64, V1 ⊗ V2 ← (V3 ⊗ V4 ⊗ V5)')
    record = TensorKit._pack_tensormap(tensor)
    @test record isa NamedTuple
    @test record.kind === :dense
    for table in (record.codomain_trees, record.domain_trees)
        @test table isa NamedTuple
        @test table.uncoupled isa Matrix{sectortype(tensor)}
        @test table.coupled isa Vector{sectortype(tensor)}
        @test table.isdual isa BitMatrix
        @test table.innerlines isa Matrix{sectortype(tensor)}
        @test table.vertices isa Matrix{Int}
        @test all(!(value isa FusionTree) for value in values(table))
        @test all(!(value isa AbstractString) for value in values(table))
    end
    @test record.pair_ids isa Matrix{Int}
    @test record.shapes isa Matrix{Int}
    @test record.data isa Vector{ComplexF64}
end

@testset "Fusion-tree iteration-order independence" begin
    V1, V2, V3, V4, V5 = TestSetup.VRepA4
    tensor = randn(ComplexF64, V1 ⊗ V2 ← (V3 ⊗ V4 ⊗ V5)')
    @test TensorKit._unpack_tensormap(reverse_record(TensorKit._pack_tensormap(tensor))) == tensor

    V = Vect[SU2Irrep](0 => 3, 1 // 2 => 2, 1 => 1)'
    diagonal = DiagonalTensorMap(randn(ComplexF64, reduceddim(V)), V)
    @test TensorKit._unpack_tensormap(reverse_record(TensorKit._pack_tensormap(diagonal))) == diagonal
end

@testset "TensorMap file validation" begin
    V = Vect[Z2Irrep](0 => 2, 1 => 3)
    tensor = randn(Float64, V ⊗ V ← V ⊗ V)
    record = TensorKit._pack_tensormap(tensor)
    mktempdir() do directory
        path = joinpath(directory, "invalid.jld2")

        write_record(path, record; format = "not TensorKit")
        @test_throws ArgumentError load_tensor(path)

        write_record(path, record; version = TensorKit.TENSORMAP_FILE_VERSION + 1)
        @test_throws ArgumentError load_tensor(path)

        JLD2.jldsave(path; unrelated = tensor.data)
        @test_throws ArgumentError load_tensor(path)

        write_record(path, merge(record, (kind = :unknown,)))
        @test_throws ArgumentError load_tensor(path)

        duplicate_ids = copy(record.pair_ids)
        duplicate_ids[:, 2] .= duplicate_ids[:, 1]
        write_record(path, merge(record, (pair_ids = duplicate_ids,)))
        @test_throws ArgumentError load_tensor(path)

        write_record(path, merge(record, (
            pair_ids = record.pair_ids[:, 1:(end - 1)],
            shapes = record.shapes[:, 1:(end - 1)],
        )))
        @test_throws ArgumentError load_tensor(path)

        invalid_ids = copy(record.pair_ids)
        invalid_ids[1, 1] = size(record.codomain_trees.coupled, 1) + 1
        write_record(path, merge(record, (pair_ids = invalid_ids,)))
        @test_throws ArgumentError load_tensor(path)

        duplicate_table = merge(record.codomain_trees, (
            uncoupled = hcat(record.codomain_trees.uncoupled, record.codomain_trees.uncoupled[:, 1]),
            coupled = vcat(record.codomain_trees.coupled, record.codomain_trees.coupled[1]),
            isdual = hcat(record.codomain_trees.isdual, record.codomain_trees.isdual[:, 1]),
            innerlines = hcat(record.codomain_trees.innerlines, record.codomain_trees.innerlines[:, 1]),
            vertices = hcat(record.codomain_trees.vertices, record.codomain_trees.vertices[:, 1]),
        ))
        write_record(path, merge(record, (codomain_trees = duplicate_table,)))
        @test_throws ArgumentError load_tensor(path)

        bad_dimensions = merge(record.codomain_trees, (
            uncoupled = record.codomain_trees.uncoupled[1:(end - 1), :],
        ))
        write_record(path, merge(record, (codomain_trees = bad_dimensions,)))
        @test_throws DimensionMismatch load_tensor(path)

        bad_vertices = copy(record.codomain_trees.vertices)
        bad_vertices[1, 1] = 2
        invalid_tree = merge(record.codomain_trees, (vertices = bad_vertices,))
        write_record(path, merge(record, (codomain_trees = invalid_tree,)))
        @test_throws ArgumentError load_tensor(path)

        incompatible_space = Vect[Z2Irrep](0 => 2) ⊗ Vect[Z2Irrep](0 => 2) ←
            Vect[Z2Irrep](0 => 2) ⊗ Vect[Z2Irrep](0 => 2)
        write_record(path, merge(record, (space = incompatible_space,)))
        @test_throws ArgumentError load_tensor(path)

        badshape = copy(record.shapes)
        badshape[1, 1] += 1
        write_record(path, merge(record, (shapes = badshape,)))
        @test_throws DimensionMismatch load_tensor(path)

        write_record(path, merge(record, (data = record.data[1:(end - 1)],)))
        @test_throws DimensionMismatch load_tensor(path)

        write_record(path, merge(record, (data = vcat(record.data, zero(eltype(record.data))),)))
        @test_throws DimensionMismatch load_tensor(path)

        write_record(path, 1)
        @test_throws ArgumentError load_tensor(path)
    end
end

@testset "TensorMap file compactness" begin
    mktempdir() do directory
        V = Vect[U1Irrep](i => 3 for i in -3:3)
        tensor = randn(ComplexF64, V ⊗ V ← V ⊗ V)
        tensor_path = joinpath(directory, "tensor.jld2")
        dict_path = joinpath(directory, "tensor-dict.jld2")
        save_tensor(tensor_path, tensor)
        JLD2.jldsave(dict_path; tensor = convert(Dict, tensor))
        tensor_size = filesize(tensor_path)
        dict_size = filesize(dict_path)
        @info "TensorMap IO size comparison" tensor_size dict_size
        @test tensor_size > 0
        @test dict_size > 0

        Vd = Vect[Z2Irrep](0 => 20, 1 => 20)
        diagonal = DiagonalTensorMap(randn(Float64, reduceddim(Vd)), Vd)
        diagonal_path = joinpath(directory, "diagonal.jld2")
        dense_diagonal_path = joinpath(directory, "dense-diagonal.jld2")
        save_tensor(diagonal_path, diagonal)
        save_tensor(dense_diagonal_path, TensorMap(diagonal))
        @test filesize(diagonal_path) < filesize(dense_diagonal_path)

        braiding = BraidingTensor(Vd, Vd)
        braiding_path = joinpath(directory, "braiding.jld2")
        dense_braiding_path = joinpath(directory, "dense-braiding.jld2")
        save_tensor(braiding_path, braiding)
        save_tensor(dense_braiding_path, TensorMap(braiding))
        @test filesize(braiding_path) < filesize(dense_braiding_path)
    end
end
