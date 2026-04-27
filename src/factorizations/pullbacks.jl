for pullback! in (
        :qr_pullback!, :lq_pullback!, :left_polar_pullback!, :right_polar_pullback!,
    )
    @eval function MAK.$pullback!(
            Δt::AbstractTensorMap, t::AbstractTensorMap, F, ΔF; kwargs...
        )
        foreachblock(Δt, t) do c, (Δb, b)
            Fc = block.(F, Ref(c))
            ΔFc = block.(ΔF, Ref(c))
            return MAK.$pullback!(Δb, b, Fc, ΔFc; kwargs...)
        end
        return Δt
    end
end
for pullback! in (:qr_null_pullback!, :lq_null_pullback!)
    @eval function MAK.$pullback!(
            Δt::AbstractTensorMap, t::AbstractTensorMap, F, ΔF; kwargs...
        )
        foreachblock(Δt, t) do c, (Δb, b)
            Fc = block(F, c)
            ΔFc = block(ΔF, c)
            return MAK.$pullback!(Δb, b, Fc, ΔFc; kwargs...)
        end
        return Δt
    end
end
_notrunc_ind(t) = SectorDict(c => Colon() for c in blocksectors(t))

for pullback! in (:svd_pullback!, :eig_pullback!, :eigh_pullback!)
    @eval function MAK.$pullback!(
            Δt::AbstractTensorMap, t::AbstractTensorMap, F, ΔF, inds = _notrunc_ind(t);
            kwargs...
        )
        foreachblock(Δt, t) do c, (Δb, b)
            haskey(inds, c) || return nothing
            ind = inds[c]
            Fc = block.(F, Ref(c))
            ΔFc = block.(ΔF, Ref(c))
            MAK.$pullback!(Δb, b, Fc, ΔFc, ind; kwargs...)
            return nothing
        end
        return Δt
    end
end

for pullback_trunc! in (:svd_trunc_pullback!, :eig_trunc_pullback!, :eigh_trunc_pullback!)
    @eval function MAK.$pullback_trunc!(
            Δt::AbstractTensorMap, t::AbstractTensorMap, F, ΔF; kwargs...
        )
        foreachblock(Δt, t) do c, (Δb, b)
            Fc = block.(F, Ref(c))
            ΔFc = block.(ΔF, Ref(c))
            MAK.$pullback_trunc!(Δb, b, Fc, ΔFc; kwargs...)
            return nothing
        end
        return Δt
    end
end

for f in (:qr, :lq)
    remove_f_gauge_dependence! = Symbol(:remove_, f, :_gauge_dependence!)
    remove_f_null_gauge_dependence! = Symbol(:remove_, f, :_null_gauge_dependence!)
    @eval function MAK.$remove_f_gauge_dependence!(
            ΔF₁::AbstractTensorMap, ΔF₂::AbstractTensorMap, A, F₁, F₂;
            kwargs...
        )
        foreachblock(ΔF₁, ΔF₂, A, F₁, F₂) do _, (Δf₁, Δf₂, a, f₁, f₂)
            MAK.$remove_f_gauge_dependence!(Δf₁, Δf₂, a, f₁, f₂)
            return nothing
        end
        return ΔF₁, ΔF₂
    end
    # Already captured by MAK implementation
    # @eval function MAK.$remove_f_null_gauge_dependence!(ΔN::AbstractTensorMap, A, N; kwargs...)
    #     foreachblock(ΔN, A, N) do _, (Δn, a, n)
    #         $remove_f_gauge_dependence!(Δn, a, n)
    #     end
    #     return ΔN
    # end
end

for f in (:eig, :eigh)
    remove_f_gauge_dependence! = Symbol(:remove_, f, :_gauge_dependence!)
    @eval function MAK.$remove_f_gauge_dependence!(
            ΔV::AbstractTensorMap, D, V, inds = _notrunc_ind(ΔV);
            kwargs...
        )
        foreachblock(ΔV, D, V) do c, (Δv, d, v)
            haskey(inds, c) || return nothing
            ind = inds[c]
            MAK.$remove_f_gauge_dependence!(Δv, d, v, ind; kwargs...)
            return nothing
        end
        return ΔV
    end
end
function MAK.remove_svd_gauge_dependence!(
        ΔU::AbstractTensorMap, ΔVᴴ::AbstractTensorMap, U, S, Vᴴ;
        kwargs...
    )
    foreachblock(ΔU, ΔVᴴ, U, S, Vᴴ) do c, (Δu, Δvᴴ, u, s, vᴴ)
        MAK.remove_svd_gauge_dependence!(Δu, Δvᴴ, u, s, vᴴ)
        return nothing
    end
    return ΔU, ΔVᴴ
end
