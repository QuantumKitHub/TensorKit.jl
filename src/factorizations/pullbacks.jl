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
function _notrunc_ind(t)
    I = sectortype(t)
    return _builddensemap(sectorstoragetype(I), I, blocks(t), Colon) do _, _
        Colon()
    end
end

for pullback! in (:svd_pullback!, :eig_pullback!, :eigh_pullback!)
    @eval function MAK.$pullback!(
            Δt::AbstractTensorMap, t::AbstractTensorMap, F, ΔF, inds = _notrunc_ind(t);
            kwargs...
        )
        Isec = sectortype(t)
        foreachblock(Δt, t) do c, (Δb, b)
            ind = _denseget(inds, Isec, c)
            isnothing(ind) && return nothing
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
    @eval function MAK.$remove_f_gauge_dependence!(
            ΔF₁::AbstractTensorMap, ΔF₂::AbstractTensorMap, A, F₁, F₂;
            kwargs...
        )
        foreachblock(ΔF₁, ΔF₂, A, F₁, F₂) do _, (Δf₁, Δf₂, a, f₁, f₂)
            MAK.$remove_f_gauge_dependence!(Δf₁, Δf₂, a, f₁, f₂; kwargs...)
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
    @eval function MAK.$remove_f_gauge_dependence!(ΔV::AbstractTensorMap, D, V; kwargs...)
        foreachblock(ΔV, D, V) do c, (Δv, d, v)
            MAK.$remove_f_gauge_dependence!(Δv, d, v; kwargs...)
            return nothing
        end
        return ΔV
    end
end
function MAK.remove_svd_gauge_dependence!(
        ΔU::AbstractTensorMap, ΔVᴴ::AbstractTensorMap, U, S, Vᴴ; kwargs...
    )
    foreachblock(ΔU, ΔVᴴ, U, S, Vᴴ) do c, (Δu, Δvᴴ, u, s, vᴴ)
        MAK.remove_svd_gauge_dependence!(Δu, Δvᴴ, u, s, vᴴ; kwargs...)
        return nothing
    end
    return ΔU, ΔVᴴ
end
