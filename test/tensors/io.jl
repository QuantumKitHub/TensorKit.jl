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

"""Write a raw TensorKit tensor record for malformed-file tests."""
function write_record(path, record; format = TensorKit.TENSORMAP_FILE_FORMAT, version = TensorKit.TENSORMAP_FILE_VERSION)
    return JLD2.jldopen(path, "w") do file
        file["format"] = format
        file["version"] = version
        file["tensor"] = record
    end
end

@testset "TensorMap save and load" begin
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
            @test save(path, tensor) === nothing
            restored = load(path)
            @test restored isa TensorMap
            @test storagetype(restored) === Vector{ComplexF64}
            @test space(restored) == space(tensor)
            @test restored == tensor
        end

        tensor = randn(Float64, ℂ^2 ⊗ ℂ^3)
        restored = load((path = joinpath(directory, "plain-tensor.jld2"); save(path, tensor); path))
        @test restored isa Tensor
        @test restored == tensor

        empty_tensor = randn(Float64, zero(ℂ^2) ← ℂ^2)
        restored_empty = load((path = joinpath(directory, "empty.jld2"); save(path, empty_tensor); path))
        @test restored_empty == empty_tensor
        @test isempty(restored_empty.data)

        source = randn(Float64, ℂ^3 ← ℂ^2)
        custom_data = TestDenseVector(copy(source.data))
        custom = TensorMap{Float64, ComplexSpace, 1, 1, typeof(custom_data)}(custom_data, space(source))
        @test storagetype(custom) === TestDenseVector{Float64}
        restored_custom = load((path = joinpath(directory, "custom.jld2"); save(path, custom); path))
        @test storagetype(restored_custom) === Vector{Float64}
        @test restored_custom == custom

        diagonal_space = Vect[SU2Irrep](0 => 3, 1 // 2 => 2, 1 => 1)'
        diagonal = DiagonalTensorMap(randn(ComplexF64, reduceddim(diagonal_space)), diagonal_space)
        restored_diagonal = load((path = joinpath(directory, "diagonal.jld2"); save(path, diagonal); path))
        @test restored_diagonal isa DiagonalTensorMap
        @test storagetype(restored_diagonal) === Vector{ComplexF64}
        @test restored_diagonal == diagonal

        braid_space = Vect[FibonacciAnyon](:I => 3, :τ => 2)
        for braiding in (BraidingTensor(braid_space, braid_space'), BraidingTensor(braid_space, braid_space')')
            path = joinpath(directory, "braiding-$(braiding.adjoint).jld2")
            save(path, braiding)
            restored_braiding = load(path)
            @test restored_braiding isa BraidingTensor
            @test storagetype(restored_braiding) === Vector{eltype(braiding)}
            @test restored_braiding.V1 == braiding.V1
            @test restored_braiding.V2 == braiding.V2
            @test restored_braiding.adjoint == braiding.adjoint
            @test TensorMap(restored_braiding) == TensorMap(braiding)
        end

        @test_throws ArgumentError save(joinpath(directory, "adjoint.jld2"), source')
        @test_throws ArgumentError save(joinpath(directory, "unsupported.jld2"), UnregisteredTensorMap())
    end
end

@testset "TensorMap file validation" begin
    tensor = randn(Float64, Vect[Z2Irrep](0 => 2, 1 => 3) ← Vect[Z2Irrep](0 => 3, 1 => 2))
    record = TensorKit._pack_tensormap(tensor)
    mktempdir() do directory
        path = joinpath(directory, "invalid.jld2")

        write_record(path, record; format = "not TensorKit")
        @test_throws ArgumentError load(path)

        write_record(path, record; version = TensorKit.TENSORMAP_FILE_VERSION + 1)
        @test_throws ArgumentError load(path)

        JLD2.jldsave(path; unrelated = tensor.data)
        @test_throws ArgumentError load(path)

        duplicate = TensorKit.DenseTensorMapRecordV1(
            record.space,
            [record.sectors[1], record.sectors[1]],
            [record.blockshapes[1], record.blockshapes[1]],
            vcat(record.data[1:prod(record.blockshapes[1])], record.data[1:prod(record.blockshapes[1])]),
        )
        write_record(path, duplicate)
        @test_throws ArgumentError load(path)

        badshape = copy(record.blockshapes)
        badshape[1] = (badshape[1][1] + 1, badshape[1][2])
        write_record(path, TensorKit.DenseTensorMapRecordV1(record.space, record.sectors, badshape, record.data))
        @test_throws DimensionMismatch load(path)

        write_record(
            path,
            TensorKit.DenseTensorMapRecordV1(record.space, record.sectors, record.blockshapes, record.data[1:(end - 1)]),
        )
        @test_throws DimensionMismatch load(path)

        write_record(path, 1)
        @test_throws ArgumentError load(path)
    end
end

@testset "TensorMap file compactness" begin
    mktempdir() do directory
        V = Vect[U1Irrep](i => 3 for i in -3:3)
        tensor = randn(ComplexF64, V ⊗ V ← V ⊗ V)
        compact_path = joinpath(directory, "tensor.jld2")
        dict_path = joinpath(directory, "tensor-dict.jld2")
        save(compact_path, tensor)
        JLD2.jldsave(dict_path; tensor = convert(Dict, tensor))
        compact_size = filesize(compact_path)
        dict_size = filesize(dict_path)
        @test compact_size < dict_size

        Vd = Vect[Z2Irrep](0 => 20, 1 => 20)
        diagonal = DiagonalTensorMap(randn(Float64, reduceddim(Vd)), Vd)
        diagonal_path = joinpath(directory, "diagonal.jld2")
        dense_diagonal_path = joinpath(directory, "dense-diagonal.jld2")
        save(diagonal_path, diagonal)
        save(dense_diagonal_path, TensorMap(diagonal))
        @test filesize(diagonal_path) < filesize(dense_diagonal_path)

        braiding = BraidingTensor(Vd, Vd)
        braiding_path = joinpath(directory, "braiding.jld2")
        dense_braiding_path = joinpath(directory, "dense-braiding.jld2")
        save(braiding_path, braiding)
        save(dense_braiding_path, TensorMap(braiding))
        @test filesize(braiding_path) < filesize(dense_braiding_path)
    end
end
