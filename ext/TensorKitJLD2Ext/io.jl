function save_tensor(path::AbstractString, tensor::AbstractTensorMap)
    record = _pack_tensormap(tensor)
    destination = abspath(path)
    mktemp(dirname(destination)) do temporary, io
        close(io)
        JLD2.jldsave(
            temporary; format = TENSORMAP_FILE_FORMAT,
            version = TENSORMAP_FILE_VERSION, tensor = record
        )
        mv(temporary, destination; force = true)
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
