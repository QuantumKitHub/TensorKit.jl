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
    @eval function MAK.$pullback!(
            Δt::AbstractTensorMap, ::Nothing, F, ΔF; kwargs...
        )
        foreachblock(Δt) do c, (Δb,)
            Fc = block.(F, Ref(c))
            ΔFc = block.(ΔF, Ref(c))
            return MAK.$pullback!(Δb, nothing, Fc, ΔFc; kwargs...)
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

for pullback! in (:eig_vals_pullback!, :eigh_vals_pullback!)
    @eval function MAK.$pullback!(
            Δt::AbstractTensorMap, ::Nothing, DV::Tuple{Diagonal, <:AbstractTensorMap}, ΔD, inds;
            kwargs...
        )
        return MAK.$pullback!(Δt, nothing, (MAK.diagonal(parent(DV[1])), DV[2]), ΔD, inds; kwargs...)
    end
end
function MAK.svd_vals_pullback!(
        Δt::AbstractTensorMap, ::Nothing, USVᴴ::Tuple{<:AbstractTensorMap, Diagonal, <:AbstractTensorMap}, ΔS, ind;
        kwargs...
    )
    return MAK.svd_vals_pullback!(Δt, nothing, (USVᴴ[1], MAK.diagonal(parent(USVᴴ[2])), USVᴴ[3]), ΔS, ind; kwargs...)
end

nothing_or_block(x, c) = isnothing(x) ? x : block(x, c)
nothing_or_block(x::Diagonal, c) = block(MAK.diagonal(parent(x)), c)
nothing_or_foreachblock(f, Δt, t) = isnothing(t) ? foreachblock(f, Δt) : foreachblock(f, Δt, t) 
for pullback! in (:svd_pullback!, :eig_pullback!, :eigh_pullback!)
    @eval function MAK.$pullback!(
            Δt::AbstractTensorMap, t, F, ΔF, inds = _notrunc_ind(Δt);
            kwargs...
        )
        nothing_or_foreachblock(Δt, t) do c, Δbb
            Δb, b = length(Δbb) == 1 ? (only(Δbb), nothing) : Δbb
            ind = get(inds, c, nothing)
            isnothing(ind) && return nothing
            Fc = nothing_or_block.(F, Ref(c))
            ΔFc = nothing_or_block.(ΔF, Ref(c))
            MAK.$pullback!(Δb, b, Fc, ΔFc, ind; kwargs...)
            return nothing
        end
        return Δt
    end
    @eval function MAK.$pullback!(
            Δt::AbstractTensorMap, t, F, ΔF, ::Colon; kwargs...
        )
        return MAK.$pullback!(Δt, t, F, ΔF, _notrunc_ind(Δt); kwargs...)
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

MAK.has_equal_storage(A::AbstractTensorMap, B::AbstractTensorMap) = A === B
MAK.has_equal_storage(A::AbstractTensorMap, B::SectorVector) = false
MAK.has_equal_storage(A::SectorVector, B::AbstractTensorMap) = false
