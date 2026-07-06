# planartrace!
# ------------
# TODO: Fix planartrace pullback
# This implementation is slightly more involved than its non-planar counterpart
# this is because we lack a general `pAB` argument in `planarcontract`, and need
# to keep things planar along the way.
# In particular, we can't simply tensor product with multiple identities in one go
# if they aren't "contiguous", e.g. p = ((1, 4, 5), ()), q = ((2, 6), (3, 7))

# @is_primitive(
#     DefaultCtx,
#     ReverseMode,
#     Tuple{
#         typeof(TensorKit.planartrace!),
#         AbstractTensorMap,
#         AbstractTensorMap, Index2Tuple, Index2Tuple,
#         Number, Number,
#         Any, Any,
#     }
# )

# function Mooncake.rrule!!(
#         ::CoDual{typeof(TensorKit.planartrace!)},
#         C_ΔC::CoDual{<:AbstractTensorMap},
#         A_ΔA::CoDual{<:AbstractTensorMap}, p_Δp::CoDual{<:Index2Tuple}, q_Δq::CoDual{<:Index2Tuple},
#         α_Δα::CoDual{<:Number}, β_Δβ::CoDual{<:Number},
#         backend_Δbackend::CoDual, allocator_Δallocator::CoDual
#     )
#     # prepare arguments
#     C, ΔC = arrayify(C_ΔC)
#     A, ΔA = arrayify(A_ΔA)
#     p = primal(p_Δp)
#     q = primal(q_Δq)
#     α, β = primal.((α_Δα, β_Δβ))
#     backend, allocator = primal.((backend_Δbackend, allocator_Δallocator))
#
#     # primal call
#     C_cache = copy(C)
#     TensorKit.planartrace!(C, A, p, q, α, β, backend, allocator)
#
#     function planartrace_pullback(::NoRData)
#         copy!(C, C_cache)
#
#         ΔAr = TK.planartrace_pullback_ΔA!(ΔA, ΔC, A, p, q, α, backend, allocator) # this typically returns nothing
#         Δαr = TK.planartrace_pullback_Δα(ΔC, A, p, q, α, backend, allocator)
#         Δβr = pullback_dβ(ΔC, C, β)
#         ΔCr = pullback_dC!(ΔC, β) # this typically returns nothing
#
#         return NoRData(),
#             NoRData(), NoRData(), NoRData(), NoRData(),
#             Δαr, Δβr, NoRData(), NoRData()
#     end
#
#     return C_ΔC, planartrace_pullback
# end

