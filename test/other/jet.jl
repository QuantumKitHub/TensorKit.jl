using TensorKit
using Test
using JET

# Reports that are not TensorKit's to fix. Each entry needs an upstream issue.
const IGNORED = (
    # `local variable kwargs may be undefined` inside `GenericTreeTransformer`, coming entirely
    # from Base's `@debug` expansion (`local msg, kwargs` bound in a short-circuit guard) rather
    # than from any TensorKit code: https://github.com/aviatesk/JET.jl/issues/860
    # `AnyFrameMethod` rather than `LastFrameMethod`: the `@debug` sits inside a `taskforeach`
    # closure, so the innermost frame is a gensym'd closure name rather than the constructor.
    JET.AnyFrameMethod(:GenericTreeTransformer),
    # `no matching method found convert(::Type{UniqueTreeTransformer}, ::GenericTreeTransformer)`
    # (and vice versa) in the uncached bodies of `treebraider` and `treetransposer`: for abstract
    # spaces, JET cannot correlate the `FusionStyle` branch of the return type annotation, which
    # `@cached` keeps on the body, with that of `TreeTransformer`.
    (
        JET.LastFrameMethod(only(methods(TensorKit.MemoizationKit.implementation, Tuple{typeof(f), Vararg{Any}})))
            for f in (TensorKit.treebraider, TensorKit.treetransposer)
    )...,
)

# also gated in runtests.jl; JET < 0.12 does not define `JET_AVAILABLE`
if isdefined(JET, :JET_AVAILABLE) && JET.JET_AVAILABLE
    JET.test_package(TensorKit; target_modules = (TensorKit,), ignored_modules = IGNORED)
else
    @info "Full JET functionality is unavailable on Julia $VERSION; skipping JET analysis"
end
