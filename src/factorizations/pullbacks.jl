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
            return MAK.$pullback_trunc!(Δb, b, Fc, ΔFc; kwargs...)
        end
        return Δt
    end
end
