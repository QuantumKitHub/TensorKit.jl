function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(MatrixAlgebraKit.copy_input)},
        ::Type{RT},
        cache,
        f::Annotation,
        A::Annotation{<:AbstractTensorMap}
    ) where {RT}
    copy_shadow = cache
    if !isa(A, Const) && !isnothing(copy_shadow)
        add!(A.dval, copy_shadow)
    end
    return (nothing, nothing)
end

for (f, pb) in (
        (:eig_full, :(MatrixAlgebraKit.eig_pullback!)),
        (:eigh_full, :(MatrixAlgebraKit.eigh_pullback!)),
        (:lq_compact, :(MatrixAlgebraKit.lq_pullback!)),
        (:qr_compact, :(MatrixAlgebraKit.qr_pullback!)),
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

# TODO
#=
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(svd_trunc)},
        ::Type{RT},
        A::Annotation{<:AbstractTensorMap},
        alg::Const,
    ) where {RT}

    USVᴴ = svd_compact(A.val, alg.val.alg)
    USVᴴtrunc, ind = MatrixAlgebraKit.truncate(svd_trunc!, USVᴴ, alg.val.trunc)
    ϵ = MatrixAlgebraKit.truncation_error(diagview(USVᴴ[2]), ind)
    dUSVᴴtrunc = make_zero(USVᴴtrunc)
    cache = (USVᴴtrunc, dUSVᴴtrunc)
    return EnzymeRules.AugmentedReturn(USVᴴtrunc, dUSVᴴtrunc, cache) 
end
function EnzymeRules.reverse(
    config::EnzymeRules.RevConfigWidth{1},
    func::Const{typeof(svd_trunc)},
    ::Type{RT},
    cache,
    A::Annotation{<:AbstractTensorMap},
    alg::Const,
) where {RT}
    USVᴴ, dUSVᴴ = cache
    MatrixAlgebraKit.svd_pullback!(A.dval, A.val, USVᴴ, dUSVᴴ)
    return (nothing, nothing) 
end=#
