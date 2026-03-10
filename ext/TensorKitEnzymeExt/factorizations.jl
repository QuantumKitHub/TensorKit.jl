# need these due to Enzyme choking on blocks

for f in (:project_hermitian, :project_antihermitian)
    f! = Symbol(f, :!)
    @eval begin
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f!)},
                ::Type{RT},
                A::Annotation{<:AbstractTensorMap},
                arg::Annotation{<:AbstractTensorMap},
                alg::Const,
            ) where {RT}
            $f!(A.val, arg.val, alg.val)
            primal = EnzymeRules.needs_primal(config) ? arg.val : nothing
            shadow = EnzymeRules.needs_shadow(config) ? arg.dval : nothing
            cache = nothing
            return EnzymeRules.AugmentedReturn(primal, shadow, cache)
        end
        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f!)},
                ::Type{RT},
                cache,
                A::Annotation{<:AbstractTensorMap},
                arg::Annotation{<:AbstractTensorMap},
                alg::Const,
            ) where {RT}
            if !isa(A, Const)
                $f!(arg.dval, arg.dval, alg.val)
                if A.dval !== arg.dval
                    A.dval .+= arg.dval
                    make_zero!(arg.dval)
                end
            end
            return (nothing, nothing, nothing)
        end
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f)},
                ::Type{RT},
                A::Annotation{<:AbstractTensorMap},
                alg::Const,
            ) where {RT}
            ret = $f(A.val, alg.val)
            dret = make_zero(ret)
            primal = EnzymeRules.needs_primal(config) ? ret : nothing
            shadow = EnzymeRules.needs_shadow(config) ? dret : nothing
            cache = dret
            return EnzymeRules.AugmentedReturn(primal, shadow, cache)
        end
        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f)},
                ::Type{RT},
                cache,
                A::Annotation{<:AbstractTensorMap},
                alg::Const,
            ) where {RT}
            dret = cache
            if !isa(A, Const)
                $f!(dret, dret, alg.val)
                add!(A.dval, dret)
            end
            make_zero!(dret)
            return (nothing, nothing)
        end
    end
end

for (f, pb) in (
        (:eig_full, :(MatrixAlgebraKit.eig_pullback!)),
        (:eigh_full, :(MatrixAlgebraKit.eigh_pullback!)),
        (:lq_compact, :(MatrixAlgebraKit.lq_pullback!)),
        (:qr_compact, :(MatrixAlgebraKit.qr_pullback!)),
        (:lq_full, :(MatrixAlgebraKit.lq_pullback!)),
        (:qr_full, :(MatrixAlgebraKit.qr_pullback!)),
        (:lq_null, :(MatrixAlgebraKit.lq_null_pullback!)),
        (:qr_null, :(MatrixAlgebraKit.qr_null_pullback!)),
    )
    @eval begin
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f)},
                ::Type{RT},
                A::Annotation{<:AbstractTensorMap},
                alg::Const,
            ) where {RT}
            ret = $f(A.val, alg.val)
            primal = EnzymeRules.needs_primal(config) ? ret : nothing
            shadow = EnzymeRules.needs_shadow(config) ? make_zero(ret) : nothing
            cache = (ret, shadow)
            return EnzymeRules.AugmentedReturn(primal, shadow, cache)
        end
        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f)},
                ::Type{RT},
                cache,
                A::Annotation{<:AbstractTensorMap},
                alg::Const,
            ) where {RT}
            !isa(A, Const) && $pb(A.dval, A.val, cache...)
            return (nothing, nothing)
        end
    end
end

for (f, f_full, pb) in (
        (:eig_vals, :eig_full, :(MatrixAlgebraKit.eig_vals_pullback!)),
        (:eigh_vals, :eigh_full, :(MatrixAlgebraKit.eigh_vals_pullback!)),
    )
    @eval begin
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f)},
                ::Type{RT},
                A::Annotation{<:AbstractTensorMap},
                alg::Const,
            ) where {RT}
            ret_full = $f_full(A.val, alg.val)
            ret = diagview(ret_full[1])
            primal = EnzymeRules.needs_primal(config) ? ret : nothing
            shadow = EnzymeRules.needs_shadow(config) ? make_zero(ret) : nothing
            cache = (ret, shadow, ret_full[2])
            return EnzymeRules.AugmentedReturn(primal, shadow, cache)
        end
        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f)},
                ::Type{RT},
                cache,
                A::Annotation{<:AbstractTensorMap},
                alg::Const,
            ) where {RT}
            D, dD, V = cache
            !isa(A, Const) && $pb(A.dval, A.val, (DiagonalTensorMap(D), V), dD)
            return (nothing, nothing)
        end
    end
end

for f in (:svd_compact, :svd_full)
    @eval begin
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f)},
                ::Type{RT},
                A::Annotation{<:AbstractTensorMap},
                alg::Const,
            ) where {RT}
            USVᴴ = $f(A.val, alg.val)
            primal = EnzymeRules.needs_primal(config) ? USVᴴ : nothing
            shadow = EnzymeRules.needs_shadow(config) ? make_zero(USVᴴ) : nothing
            cache = (USVᴴ, shadow)
            return EnzymeRules.AugmentedReturn(primal, shadow, cache)
        end
        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f)},
                ::Type{RT},
                cache,
                A::Annotation{<:AbstractTensorMap},
                alg::Const,
            ) where {RT}
            !isa(A, Const) && MatrixAlgebraKit.svd_pullback!(A.dval, A.val, cache...)
            return (nothing, nothing)
        end
    end

    # mutating version is not guaranteed to actually mutate
    # so we can simply use the non-mutating version instead
    f! = Symbol(f, :!)
    #=@eval begin
        function EnzymeRules.augmented_primal(
            config::EnzymeRules.RevConfigWidth{1},
            func::Const{typeof($f!)},
            ::Type{RT},
            A::Annotation{<:AbstractTensorMap},
            USVᴴ::Annotation,
            alg::Const,
        ) where {RT}
            EnzymeRules.augmented_primal(func, RT, A, alg)
        end
        function EnzymeRules.reverse(
            config::EnzymeRules.RevConfigWidth{1},
            func::Const{typeof($f!)},
            ::Type{RT},
            cache,
            A::Annotation{<:AbstractTensorMap},
            USVᴴ::Annotation,
            alg::Const,
        ) where {RT}
            EnzymeRules.reverse(func, RT, A, alg)
        end
    end=# #hmmmm
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(svd_trunc_no_error)},
        ::Type{RT},
        A::Annotation{<:AbstractTensorMap},
        alg::Const,
    ) where {RT}
    USVᴴ = svd_compact(A.val, alg.val.alg)
    USVᴴtrunc, ind = MatrixAlgebraKit.truncate(svd_trunc!, USVᴴ, alg.val.trunc)
    dUSVᴴtrunc = make_zero(USVᴴtrunc)
    cache = (USVᴴ, USVᴴtrunc, dUSVᴴtrunc, ind)
    return EnzymeRules.AugmentedReturn(USVᴴtrunc, dUSVᴴtrunc, cache)
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(svd_trunc_no_error)},
        ::Type{RT},
        cache,
        A::Annotation{<:AbstractTensorMap},
        alg::Const,
    ) where {RT}
    USVᴴ, USVᴴtrunc, dUSVᴴtrunc, ind = cache
    MatrixAlgebraKit.svd_pullback!(A.dval, A.val, USVᴴ, dUSVᴴtrunc, ind)
    return (nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(eig_trunc_no_error)},
        ::Type{RT},
        A::Annotation{<:AbstractTensorMap},
        alg::Const,
    ) where {RT}
    DV = eig_full(A.val, alg.val.alg)
    DVtrunc, ind = MatrixAlgebraKit.truncate(eig_trunc!, DV, alg.val.trunc)
    dDVtrunc = make_zero(DVtrunc)
    cache = (DV, DVtrunc, dDVtrunc, ind)
    return EnzymeRules.AugmentedReturn(DVtrunc, dDVtrunc, cache)
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(eig_trunc_no_error)},
        ::Type{RT},
        cache,
        A::Annotation{<:AbstractTensorMap},
        alg::Const,
    ) where {RT}
    DV, DVtrunc, dDVtrunc, ind = cache
    MatrixAlgebraKit.eig_pullback!(A.dval, A.val, DV, dDVtrunc, ind)
    return (nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(eigh_trunc_no_error)},
        ::Type{RT},
        A::Annotation{<:AbstractTensorMap},
        alg::Const,
    ) where {RT}
    DV = eigh_full(A.val, alg.val.alg)
    DVtrunc, ind = MatrixAlgebraKit.truncate(eigh_trunc!, DV, alg.val.trunc)
    dDVtrunc = make_zero(DVtrunc)
    cache = (DV, DVtrunc, dDVtrunc, ind)
    return EnzymeRules.AugmentedReturn(DVtrunc, dDVtrunc, cache)
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(eigh_trunc_no_error)},
        ::Type{RT},
        cache,
        A::Annotation{<:AbstractTensorMap},
        alg::Const,
    ) where {RT}
    DV, DVtrunc, dDVtrunc, ind = cache
    MatrixAlgebraKit.eigh_pullback!(A.dval, A.val, DV, dDVtrunc, ind)
    return (nothing, nothing)
end
