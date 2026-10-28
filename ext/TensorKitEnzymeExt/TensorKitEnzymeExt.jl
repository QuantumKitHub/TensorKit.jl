module TensorKitEnzymeExt

using Enzyme
using TensorKit
import TensorKit as TK
using TensorKit: subblock, block
using VectorInterface
using TensorOperations: TensorOperations, IndexTuple, Index2Tuple, linearize
import TensorOperations as TO
using MatrixAlgebraKit
using TupleTools
using Random: AbstractRNG

"""
    is_inactive(config, x::Annotation) -> Bool

Check whether the annotated argument `x` should be treated as inactive, i.e. whether no tangent should be read from or accumulated into `x.dval`.
Under runtime activity, Enzyme may pass an argument that is inactive at run time as a `Duplicated` whose shadow aliases the primal (`x.dval === x.val`);
writing into that shadow would then corrupt the primal.
See https://enzymead.github.io/Enzyme.jl/dev/faq/#faq-runtime-activity.

!!! note
    This is a stopgap until an equivalent helper is available in `EnzymeRules`
    (see https://github.com/EnzymeAD/Enzyme.jl/pull/3597#discussion_r4014313289).
"""
@inline is_inactive(config, ::Const) = true
@inline is_inactive(config, x::Annotation) = EnzymeRules.runtime_activity(config) && x.dval === x.val

include("utility.jl")
include("linalg.jl")
include("indexmanipulations.jl")
include("tensoroperations.jl")

end
