for f in (:svd_compact, :svd_full)
    f_pullback = Symbol(f, :_pullback)
    @eval begin
        @is_primitive DefaultCtx ReverseMode Tuple{typeof($f), AbstractTensorMap, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(::CoDual{typeof($f)}, A_dA::CoDual{<:AbstractTensorMap}, alg_dalg::CoDual)
            A, dA = arrayify(A_dA)
            alg = primal(alg_dalg)

            USVᴴ = $f(A, primal(alg_dalg))
            USVᴴ_dUSVᴴ = Mooncake.zero_fcodual(USVᴴ)
            dUSVᴴ = last.(arrayify.(USVᴴ, tangent(USVᴴ_dUSVᴴ)))

            function $f_pullback(::NoRData)
                MatrixAlgebraKit.svd_pullback!(dA, A, USVᴴ, dUSVᴴ)
                MatrixAlgebraKit.zero!.(dUSVᴴ)
                return ntuple(Returns(NoRData()), 3)
            end

            return USVᴴ_dUSVᴴ, $f_pullback
        end
    end

    # mutating version is not guaranteed to actually mutate
    # so we can simply use the non-mutating version instead and avoid having to worry about
    # storing copies and restoring state
    f! = Symbol(f, :!)
    f!_pullback = Symbol(f!, :_pullback)
    @eval begin
        @is_primitive DefaultCtx ReverseMode Tuple{typeof($f!), AbstractTensorMap, Any, MatrixAlgebraKit.AbstractAlgorithm}
        Mooncake.rrule!!(::CoDual{typeof($f!)}, A_dA::CoDual{<:AbstractTensorMap}, USVᴴ_dUSVᴴ::CoDual, alg_dalg::CoDual) =
            Mooncake.rrule!!(Mooncake.zero_fcodual($f), A_dA, alg_dalg)
    end
end

@is_primitive DefaultCtx ReverseMode Tuple{typeof(svd_trunc), AbstractTensorMap, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(
        ::CoDual{typeof(svd_trunc)},
        A_dA::CoDual{<:AbstractTensorMap},
        alg_dalg::CoDual{<:MatrixAlgebraKit.TruncatedAlgorithm}
    )
    A, dA = arrayify(A_dA)
    alg = primal(alg_dalg)

    USVᴴ = svd_compact(A, alg.alg)
    USVᴴtrunc, ind = MatrixAlgebraKit.truncate(svd_trunc!, USVᴴ, alg.trunc)
    ϵ = MatrixAlgebraKit.truncation_error(diagview(USVᴴ[2]), ind)

    USVᴴtrunc_dUSVᴴtrunc = Mooncake.zero_fcodual((USVᴴtrunc..., ϵ))
    dUSVᴴtrunc = last.(arrayify.(USVᴴtrunc, Base.front(tangent(USVᴴtrunc_dUSVᴴtrunc))))

    function svd_trunc_pullback((_, _, _, dϵ)::Tuple{NoRData, NoRData, NoRData, Real})
        abs(dϵ) ≤ MatrixAlgebraKit.defaulttol(dϵ) ||
            @warn "Gradient for `svd_trunc` ignores non-zero tangents for truncation error"
        MatrixAlgebraKit.svd_pullback!(dA, A, USVᴴ, dUSVᴴtrunc, ind)
        return ntuple(Returns(NoRData()), 3)
    end

    return USVᴴtrunc_dUSVᴴtrunc, svd_trunc_pullback
end

@is_primitive DefaultCtx ReverseMode Tuple{typeof(svd_trunc!), AbstractTensorMap, Any, MatrixAlgebraKit.AbstractAlgorithm}
Mooncake.rrule!!(::CoDual{typeof(svd_trunc!)}, A_dA::CoDual{<:AbstractTensorMap}, USVᴴ_dUSVᴴ::CoDual, alg_dalg::CoDual) =
    Mooncake.rrule!!(Mooncake.zero_fcodual(svd_trunc), A_dA, alg_dalg)
