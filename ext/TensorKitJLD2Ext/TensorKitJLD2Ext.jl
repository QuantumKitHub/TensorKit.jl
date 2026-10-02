module TensorKitJLD2Ext

using TensorKit
using TensorKit: AdjointTensorMap
import TensorKit: save_tensor, load_tensor
import JLD2

# TensorMap IO
#=============#

const TENSORMAP_FILE_FORMAT = "TensorKit.AbstractTensorMap"
const TENSORMAP_FILE_VERSION = UInt16(1)

include("fusiontrees.jl")
include("records.jl")
include("io.jl")

end
