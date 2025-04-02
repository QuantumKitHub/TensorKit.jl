using Zygote, TensorKit

_safe_pow(a::Real, pow::Real, tol::Real) = (pow < 0 && abs(a) < tol) ? zero(a) : a^pow

# Element-wise multiplication of TensorMaps respecting block structure
function _elementwise_mult(a₁::AbstractTensorMap, a₂::AbstractTensorMap)
    dst = similar(a₁)
    for (k, b) in blocks(dst)
        copyto!(b, block(a₁, k) .* block(a₂, k))
    end
    return dst
end
"""
    sdiag_pow(s, pow::Real; tol::Real=eps(scalartype(s))^(3 / 4))

Compute `s^pow` for a diagonal matrix `s`.
"""
function sdiag_pow(s::DiagonalTensorMap, pow::Real; tol::Real=eps(scalartype(s))^(3 / 4))
    # Relative tol w.r.t. largest singular value (use norm(∘, Inf) to make differentiable)
    tol *= norm(s, Inf)
    spow = DiagonalTensorMap(_safe_pow.(s.data, pow, tol), space(s, 1))
    return spow
end
function sdiag_pow(s::AbstractTensorMap{T,S,1,1}, pow::Real;
                   tol::Real=eps(scalartype(s))^(3 / 4)) where {T,S}
    # Relative tol w.r.t. largest singular value (use norm(∘, Inf) to make differentiable)
    tol *= norm(s, Inf)
    spow = similar(s)
    for (k, b) in blocks(s)
        copyto!(block(spow, k),
                LinearAlgebra.diagm(_safe_pow.(LinearAlgebra.diag(b), pow, tol)))
    end
    return spow
end

function ChainRulesCore.rrule(::typeof(sdiag_pow),
                              s::AbstractTensorMap,
                              pow::Real;
                              tol::Real=eps(scalartype(s))^(3 / 4),)
    tol *= norm(s, Inf)
    spow = sdiag_pow(s, pow; tol)
    spow_minus1_conj = scale!(sdiag_pow(s', pow - 1; tol), pow)
    function sdiag_pow_pullback(c̄_)
        c̄ = unthunk(c̄_)
        return (ChainRulesCore.NoTangent(), _elementwise_mult(c̄, spow_minus1_conj))
    end
    return spow, sdiag_pow_pullback
end

function svd_fixed_point(A, U, S, V)
    S⁻¹ = sdiag_pow(S, -1)
    return (A * V' * S⁻¹ - U, DiagonalTensorMap(U' * A * V' * S⁻¹) - one(S),
            S⁻¹ * U' * A - V)
end

using Zygote

V = ComplexSpace(3)^2
A = randn(ComplexF64, V, V)
U, S, V = tsvd(A)

Zygote.gradient(A, U, S, V) do A, U, S, V
    du, ds, dv = svd_fixed_point(A, U, S, V)
    return norm(du) + norm(ds) + norm(dv)
end
