# Shared
# ------
pullback_dC!(ΔC, β) = (scale!(ΔC, conj(β)); return NoRData())
pullback_dβ(ΔC, C, β) = _needs_tangent(β) ? project_scalar(β, inner(C, ΔC)) : NoRData()

@is_primitive DefaultCtx Tuple{typeof(mul!), AbstractTensorMap, AbstractTensorMap, AbstractTensorMap, Number, Number}

function Mooncake.rrule!!(
        ::CoDual{typeof(mul!)},
        C_ΔC::CoDual{<:AbstractTensorMap}, A_ΔA::CoDual{<:AbstractTensorMap}, B_ΔB::CoDual{<:AbstractTensorMap},
        α_Δα::CoDual{<:Number}, β_Δβ::CoDual{<:Number}
    )
    (C, ΔC), (A, ΔA), (B, ΔB) = arrayify.((C_ΔC, A_ΔA, B_ΔB))
    α, β = primal.((α_Δα, β_Δβ))

    # primal call
    C_cache = copy(C)
    AB = if _needs_tangent(α)
        AB = A * B
        add!(C, AB, α, β)
        AB
    else
        mul!(C, A, B, α, β)
        nothing
    end

    function mul_pullback(::NoRData)
        copy!(C, C_cache)

        project_mul!(ΔA, ΔC, B', conj(α))
        project_mul!(ΔB, A', ΔC, conj(α))
        ΔAr = NoRData()
        ΔBr = NoRData()
        Δαr = isnothing(AB) ? NoRData() : project_scalar(α, inner(AB, ΔC))
        Δβr = pullback_dβ(ΔC, C, β)
        ΔCr = pullback_dC!(ΔC, β)

        return NoRData(), ΔCr, ΔAr, ΔBr, Δαr, Δβr
    end

    return C_ΔC, mul_pullback
end
function Mooncake.frule!!(
        ::Dual{typeof(mul!)},
        C_ΔC::Dual{<:AbstractTensorMap}, A_ΔA::Dual{<:AbstractTensorMap}, B_ΔB::Dual{<:AbstractTensorMap},
        α_Δα::Dual{<:Number}, β_Δβ::Dual{<:Number}
    )
    (C, ΔC), (A, ΔA), (B, ΔB) = arrayify.((C_ΔC, A_ΔA, B_ΔB))
    α, Δα = Mooncake.extract(α_Δα)
    β, Δβ = Mooncake.extract(β_Δβ)
    # ΔC′ = ΔC*β + C*Δβ + A*B*Δα + ΔA*B*α + A*ΔB*α
    scale!(ΔC, β)
    if !isa(Δβ, Mooncake.NoTangent)
        add!(ΔC, C, Δβ)
    end
    if !isa(Δα, Mooncake.NoTangent)
        project_mul!(ΔC, A, B, Δα)
    end
    project_mul!(ΔC, ΔA, B, α)
    project_mul!(ΔC, A, ΔB, α)
    mul!(C, A, B, α, β)
    return C_ΔC
end

@is_primitive DefaultCtx Tuple{typeof(norm), AbstractTensorMap, Real}
function Mooncake.rrule!!(::CoDual{typeof(norm)}, tΔt::CoDual{<:AbstractTensorMap}, pdp::CoDual{<:Real})
    t, Δt = arrayify(tΔt)
    p = primal(pdp)
    p == 2 || error("currently only implemented for p = 2")
    n = norm(t, p)
    function norm_pullback(Δn)
        x = (Δn' + Δn) / 2 / hypot(n, eps(one(n)))
        add!(Δt, t, x)
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(n, Mooncake.NoFData()), norm_pullback
end
function Mooncake.frule!!(::Dual{typeof(norm)}, tΔt::Dual{<:AbstractTensorMap}, pdp::Dual{<:Real})
    t, Δt = arrayify(tΔt)
    p, Δp = Mooncake.extract(pdp)
    p == 2 || error("currently only implemented for p = 2")
    n = norm(t, p)
    Δn = real(dot(t, Δt)) * pinv(n)
    return Dual(n, Δn)
end

@is_primitive DefaultCtx Tuple{typeof(tr), AbstractTensorMap}
function Mooncake.rrule!!(::CoDual{typeof(tr)}, A_ΔA::CoDual{<:AbstractTensorMap})
    A, ΔA = arrayify(A_ΔA)
    trace = tr(A)

    function tr_pullback(Δtrace)
        for (_, b) in blocks(ΔA)
            TensorKit.diagview(b) .+= Δtrace
        end
        return NoRData(), NoRData()
    end

    return CoDual(trace, Mooncake.NoFData()), tr_pullback
end
function Mooncake.frule!!(::Dual{typeof(tr)}, A_ΔA::Dual{<:AbstractTensorMap})
    A, ΔA = arrayify(A_ΔA)
    return Dual(tr(A), tr(ΔA))
end

@is_primitive DefaultCtx Tuple{typeof(inv), AbstractTensorMap}

function Mooncake.rrule!!(::CoDual{typeof(inv)}, A_ΔA::CoDual{<:AbstractTensorMap})
    A, ΔA = arrayify(A_ΔA)
    Ainv_ΔAinv = Mooncake.zero_fcodual(inv(A))
    Ainv, ΔAinv = arrayify(Ainv_ΔAinv)

    function inv_pullback(::NoRData)
        mul!(ΔA, Ainv' * ΔAinv, Ainv', -1, One())
        return NoRData(), NoRData()
    end

    return Ainv_ΔAinv, inv_pullback
end
function Mooncake.frule!!(::Dual{typeof(inv)}, A_ΔA::Dual{<:AbstractTensorMap})
    A, ΔA = arrayify(A_ΔA)
    Ainv = inv(A)
    ΔAinv = scale!(Ainv * ΔA * Ainv, -1)
    return Dual(Ainv, ΔAinv)
end

# single-output projections: project_hermitian!, project_antihermitian!
for (f!, f, adj) in (
        (:project_hermitian!, :project_hermitian, :project_hermitian_adjoint),
        (:project_antihermitian!, :project_antihermitian, :project_antihermitian_adjoint),
    )
    @eval begin
        @is_primitive DefaultCtx Tuple{typeof($f!), AbstractTensorMap, AbstractTensorMap, MatrixAlgebraKit.AbstractAlgorithm}
        @is_primitive DefaultCtx Tuple{typeof($f), AbstractTensorMap, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(f_df::CoDual{typeof($f!)}, A_dA::CoDual{<:AbstractTensorMap}, arg_darg::CoDual, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm})
            A, dA = arrayify(A_dA)
            arg, darg = A_dA === arg_darg ? (A, dA) : arrayify(arg_darg)

            # don't need to copy/restore A since projections don't mutate input
            argc = copy(arg)
            arg = $f!(A, arg, Mooncake.primal(alg_dalg))

            function $adj(::NoRData)
                $f!(darg)
                if dA !== darg
                    add!(dA, darg)
                    MatrixAlgebraKit.zero!(darg)
                end
                copy!(arg, argc)
                return ntuple(Returns(NoRData()), 4)
            end

            return arg_darg, $adj
        end
        function Mooncake.frule!!(f_df::Dual{typeof($f!)}, A_dA::Dual{<:AbstractTensorMap}, arg_darg::Dual, alg_dalg::Dual{<:MatrixAlgebraKit.AbstractAlgorithm})
            A, dA = arrayify(A_dA)
            arg, darg = A_dA === arg_darg ? (A, dA) : arrayify(arg_darg)
            arg = $f!(A, arg, Mooncake.primal(alg_dalg))
            $f!(dA, darg, Mooncake.primal(alg_dalg))
            return arg_darg
        end
        function Mooncake.rrule!!(f_df::CoDual{typeof($f)}, A_dA::CoDual{<:AbstractTensorMap}, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm})
            A, dA = arrayify(A_dA)
            output = $f(A, Mooncake.primal(alg_dalg))
            output_doutput = Mooncake.zero_fcodual(output)

            doutput = last(arrayify(output_doutput))
            function $adj(::NoRData)
                # TODO: need accumulating projection to avoid intermediate here
                add!(dA, $f(doutput))
                MatrixAlgebraKit.zero!(doutput)
                return ntuple(Returns(NoRData()), 3)
            end

            return output_doutput, $adj
        end
        function Mooncake.frule!!(f_df::Dual{typeof($f)}, A_dA::Dual{<:AbstractTensorMap}, alg_dalg::Dual{<:MatrixAlgebraKit.AbstractAlgorithm})
            A, dA = arrayify(A_dA)
            output = $f(A, Mooncake.primal(alg_dalg))
            doutput = $f(dA, Mooncake.primal(alg_dalg))
            return Dual(output, doutput)
        end
    end
end
