# TensorMap IO interface
#=======================#

"""
    save_tensor(path::AbstractString, tensor::AbstractTensorMap)

Save one materialized tensor map to `path` using TensorKit's versioned JLD2 format.
Numerical data is copied to CPU storage, and an existing file is replaced.

Requires loading JLD2 with `using JLD2` to activate the `TensorKitJLD2Ext` extension.
"""
function save_tensor end

"""
    load_tensor(path::AbstractString) -> AbstractTensorMap

Load one tensor map saved with [`save_tensor`](@ref), using CPU storage for numerical data.

Requires loading JLD2 with `using JLD2` to activate the `TensorKitJLD2Ext` extension.
"""
function load_tensor end
