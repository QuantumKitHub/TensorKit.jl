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
