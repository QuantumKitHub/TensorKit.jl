using TensorKit
using Test
using JET

# Reports that are not TensorKit's to fix. Each entry needs an upstream issue.
const IGNORED = (
    # `schur_full`/`schur_vals` route `Diagonal` inputs to `DiagonalAlgorithm`, which schur does
    # not implement, so `schur_full(::DiagonalTensorMap)` throws a `MethodError`. This reproduces
    # with a plain `LinearAlgebra.Diagonal` and no TensorKit involved:
    # https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/issues/276
    JET.LastFrameMethod(:schur_full!),
    JET.LastFrameMethod(:schur_vals!),
    # `local variable kwargs may be undefined` inside `GenericTreeTransformer`, coming entirely
    # from Base's `@debug` expansion (`local msg, kwargs` bound in a short-circuit guard) rather
    # than from any TensorKit code: https://github.com/aviatesk/JET.jl/issues/860
    JET.LastFrameMethod(:GenericTreeTransformer),
)

# also gated in runtests.jl; JET < 0.12 does not define `JET_AVAILABLE`
if isdefined(JET, :JET_AVAILABLE) && JET.JET_AVAILABLE
    JET.test_package(TensorKit; target_modules = (TensorKit,), ignored_modules = IGNORED)
else
    @info "Full JET functionality is unavailable on Julia $VERSION; skipping JET analysis"
end
