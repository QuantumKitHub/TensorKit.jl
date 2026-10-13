for pushforward! in (
        :qr_pushforward!, :lq_pushforward!, :left_polar_pushforward!, :right_polar_pushforward!,
    )
    @eval function MAK.$pushforward!(
            Δt::AbstractTensorMap, t, F, ΔF; kwargs...
        )
        foreachblock(Δt) do c, (Δb,)
            b = nothing_or_block(t, c)
            Fc = block.(F, Ref(c))
            ΔFc = block.(ΔF, Ref(c))
            return MAK.$pushforward!(Δb, b, Fc, ΔFc; kwargs...)
        end
        return Δt
    end
end
for pushforward! in (:qr_null_pushforward!, :lq_null_pushforward!)
    @eval function MAK.$pushforward!(
            Δt::AbstractTensorMap, t::AbstractTensorMap, F, ΔF; kwargs...
        )
        foreachblock(Δt, t) do c, (Δb, b)
            Fc = block(F, c)
            ΔFc = block(ΔF, c)
            return MAK.$pushforward!(Δb, b, Fc, ΔFc; kwargs...)
        end
        return Δt
    end
end

for pushforward! in (:eig_vals_pushforward!, :eigh_vals_pushforward!)
    @eval function MAK.$pushforward!(
            Δt::AbstractTensorMap, t, DV::Tuple{Diagonal, <:AbstractTensorMap}, ΔD, inds;
            kwargs...
        )
        return MAK.$pushforward!(Δt, t, (MAK.diagonal(parent(DV[1])), DV[2]), ΔD, inds; kwargs...)
    end
end
function MAK.svd_vals_pushforward!(
        Δt::AbstractTensorMap, t, USVᴴ::Tuple{<:AbstractTensorMap, Diagonal, <:AbstractTensorMap}, ΔS, ind;
        kwargs...
    )
    return MAK.svd_vals_pushforward!(Δt, t, (USVᴴ[1], MAK.diagonal(parent(USVᴴ[2])), USVᴴ[3]), ΔS, ind; kwargs...)
end

for pushforward! in (:svd_pushforward!, :eig_pushforward!, :eigh_pushforward!)
    @eval function MAK.$pushforward!(
            Δt::AbstractTensorMap, t, F, ΔF;
            kwargs...
        )
        foreachblock(Δt, t) do c, (Δb, b)
            Fc = nothing_or_block.(F, Ref(c))
            ΔFc = nothing_or_block.(ΔF, Ref(c))
            MAK.$pushforward!(Δb, b, Fc, ΔFc; kwargs...)
            return nothing
        end
        return Δt
    end
end
