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
            if !isa(A, Const) && !isa(arg, Const)
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

# Enzyme seems to have trouble with this one
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(svd_compact)},
        ::Type{RT},
        A::Annotation{<:AbstractTensorMap},
        alg::Const,
    ) where {RT}
    USVᴴ = svd_compact(A.val, alg.val)
    primal = EnzymeRules.needs_primal(config) ? USVᴴ : nothing
    shadow = EnzymeRules.needs_shadow(config) ? make_zero(USVᴴ) : nothing
    cache = (USVᴴ, shadow)
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(svd_compact)},
        ::Type{RT},
        cache,
        A::Annotation{<:AbstractTensorMap},
        alg::Const,
    ) where {RT}
    !isa(A, Const) && MatrixAlgebraKit.svd_pullback!(A.dval, A.val, cache...)
    return (nothing, nothing)
end

# Enzyme seems to have trouble with this one
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
