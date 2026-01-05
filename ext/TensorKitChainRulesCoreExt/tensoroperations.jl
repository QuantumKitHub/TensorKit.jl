# To avoid computing rrules for α and β when these aren't needed, we want to have a
# type-stable quick bail-out
_needs_tangent(x) = _needs_tangent(typeof(x))
_needs_tangent(::Type{<:Number}) = true
_needs_tangent(::Type{<:Integer}) = false
_needs_tangent(::Type{<:Union{One, Zero}}) = false

function ChainRulesCore.rrule(
        ::typeof(TensorOperations.tensoradd!),
        C::AbstractTensorMap,
        A::AbstractTensorMap, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number, ba...
    )
    C′ = tensoradd!(copy(C), A, pA, conjA, α, β, ba...)

    projectA = ProjectTo(A)
    projectC = ProjectTo(C)
    projectα = ProjectTo(α)
    projectβ = ProjectTo(β)

    function pullback(ΔC′)
        ΔC = unthunk(ΔC′)
        dC = β === Zero() ? ZeroTangent() : @thunk projectC(scale(ΔC, conj(β)))
        dA = @thunk let
            ipA = invperm(linearize(pA))
            pdA = _repartition(ipA, A)
            TA = promote_add(ΔC, α)
            # TODO: allocator
            _dA = tensoralloc_add(TA, ΔC, pdA, conjA, Val(false))
            _dA = tensoradd!(_dA, ΔC, pdA, conjA, conjA ? α : conj(α), Zero(), ba...)
            projectA(_dA)
        end
        dα = if _needs_tangent(α)
            @thunk let
                # TODO: this is an inner product implemented as a contraction
                # for non-symmetric tensors this might be more efficient like this,
                # but for symmetric tensors an intermediate object will anyways be created
                # and then it might be more efficient to use an addition and inner product
                tΔC = _twist_nocopy(ΔC, filter(x -> isdual(space(ΔC, x)), allind(ΔC)))
                _dα = tensorscalar(
                    tensorcontract(
                        A, ((), linearize(pA)), !conjA,
                        tΔC, (trivtuple(TO.numind(pA)), ()), false,
                        ((), ()), One(), ba...
                    )
                )
                projectα(_dα)
            end
        else
            ZeroTangent()
        end
        dβ = _needs_tangent(β) ? @thunk(projectβ(inner(C, ΔC))) : ZeroTangent()
        dba = map(_ -> NoTangent(), ba)
        return NoTangent(), dC, dA, NoTangent(), NoTangent(), dα, dβ, dba...
    end

    return C′, pullback
end

function ChainRulesCore.rrule(
        ::typeof(TensorOperations.tensorcontract!),
        C::AbstractTensorMap,
        A::AbstractTensorMap, pA::Index2Tuple, conjA::Bool,
        B::AbstractTensorMap, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number, ba...
    )
    C′ = tensorcontract!(copy(C), A, pA, conjA, B, pB, conjB, pAB, α, β, ba...)

    projectA = ProjectTo(A)
    projectB = ProjectTo(B)
    projectC = ProjectTo(C)
    projectα = ProjectTo(α)
    projectβ = ProjectTo(β)

    function pullback(ΔC′)
        ΔC = unthunk(ΔC′)
        ipAB = invperm(linearize(pAB))
        pΔC = _repartition(ipAB, TO.numout(pA))

        dC = β === Zero() ? ZeroTangent() : @thunk projectC(scale(ΔC, conj(β)))
        dA = @thunk let
            ipA = _repartition(invperm(linearize(pA)), A)
            conjΔC = conjA
            conjB′ = conjA ? conjB : !conjB
            TA = promote_contract(scalartype(ΔC), scalartype(B), scalartype(α))
            # TODO: allocator
            tB = _twist_nocopy(
                B,
                TupleTools.vcat(
                    filter(x -> !isdual(space(B, x)), pB[1]),
                    filter(x -> isdual(space(B, x)), pB[2])
                )
            )
            _dA = tensoralloc_contract(
                TA, ΔC, pΔC, conjΔC, tB, reverse(pB), conjB′, ipA, Val(false)
            )
            _dA = tensorcontract!(
                _dA,
                ΔC, pΔC, conjΔC,
                tB, reverse(pB), conjB′,
                ipA,
                conjA ? α : conj(α), Zero(), ba...
            )
            projectA(_dA)
        end
        dB = @thunk let
            ipB = _repartition(invperm(linearize(pB)), B)
            conjΔC = conjB
            conjA′ = conjB ? conjA : !conjA
            TB = promote_contract(scalartype(ΔC), scalartype(A), scalartype(α))
            # TODO: allocator
            tA = _twist_nocopy(
                A,
                TupleTools.vcat(
                    filter(x -> isdual(space(A, x)), pA[1]),
                    filter(x -> !isdual(space(A, x)), pA[2])
                )
            )
            _dB = tensoralloc_contract(
                TB, tA, reverse(pA), conjA′, ΔC, pΔC, conjΔC, ipB, Val(false)
            )
            _dB = tensorcontract!(
                _dB,
                tA, reverse(pA), conjA′,
                ΔC, pΔC, conjΔC,
                ipB,
                conjB ? α : conj(α), Zero(), ba...
            )
            projectB(_dB)
        end
        dα = if _needs_tangent(α)
            @thunk let
                # TODO: this result should be AB = (C′ - βC) / α as C′ = βC + αAB
                AB = tensorcontract(A, pA, conjA, B, pB, conjB, pAB, One(), ba...)
                projectα(inner(AB, ΔC))
            end
        else
            ZeroTangent()
        end
        dβ = _needs_tangent(β) ? @thunk(projectβ(inner(C, ΔC))) : ZeroTangent()
        dba = map(_ -> NoTangent(), ba)
        return NoTangent(), dC,
            dA, NoTangent(), NoTangent(),
            dB, NoTangent(), NoTangent(),
            NoTangent(),
            dα, dβ, dba...
    end
    return C′, pullback
end

function ChainRulesCore.rrule(
        ::typeof(TensorOperations.tensortrace!),
        C::AbstractTensorMap,
        A::AbstractTensorMap, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, β::Number, ba...
    )
    C′ = tensortrace!(copy(C), A, p, q, conjA, α, β, ba...)

    projectA = ProjectTo(A)
    projectC = ProjectTo(C)
    projectα = ProjectTo(α)
    projectβ = ProjectTo(β)

    function pullback(ΔC′)
        ΔC = unthunk(ΔC′)
        dC = β === Zero() ? ZeroTangent() : @thunk projectC(scale(ΔC, conj(β)))
        dA = @thunk let
            ip = invperm((linearize(p)..., q[1]..., q[2]...))
            pdA = _repartition(ip, A)
            E = one!(TO.tensoralloc_add(scalartype(A), A, q, conjA))
            twist!(E, filter(x -> !isdual(space(E, x)), codomainind(E)))
            pE = ((), trivtuple(TO.numind(q)))
            pΔC = (trivtuple(TO.numind(p)), ())
            TA = promote_scale(ΔC, α)
            # TODO: allocator
            _dA = tensoralloc_contract(TA, ΔC, pΔC, conjA, E, pE, conjA, pdA, Val(false))
            _dA = tensorproduct!(
                _dA, ΔC, pΔC, conjA, E, pE, conjA, pdA, conjA ? α : conj(α), Zero(), ba...
            )
            projectA(_dA)
        end
        dα = if _needs_tangent(α)
            @thunk let
                # TODO: this result might be easier to compute as:
                # C′ = βC + α * trace(A) ⟹ At = (C′ - βC) / α
                At = tensortrace(A, p, q, conjA)
                projectα(inner(At, ΔC))
            end
        else
            ZeroTangent()
        end
        dβ = _needs_tangent(β) ? @thunk(projectβ(inner(C, ΔC))) : ZeroTangent()
        dba = map(_ -> NoTangent(), ba)
        return NoTangent(), dC, dA, NoTangent(), NoTangent(), NoTangent(), dα, dβ, dba...
    end

    return C′, pullback
end

function ChainRulesCore.rrule(::typeof(TensorKit.scalar), t::AbstractTensorMap)
    val = scalar(t)
    function scalar_pullback(Δval)
        dt = similar(t)
        first(blocks(dt))[2][1] = unthunk(Δval)
        return NoTangent(), dt
    end
    return val, scalar_pullback
end

# temporary function to avoid copies when not needed
# TODO: remove once `twist(t; copy=false)` is defined
function _twist_nocopy(t, inds; kwargs...)
    (BraidingStyle(sectortype(t)) isa Bosonic || isempty(inds)) && return t
    return twist(t, inds; kwargs...)
end
