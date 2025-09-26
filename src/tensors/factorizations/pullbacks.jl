for pullback! in (:qr_compact_pullback!, :lq_compact_pullback!,
                  :svd_compact_pullback!,
                  :left_polar_pullback!, :right_polar_pullback!,
                  :eig_full_pullback!, :eigh_full_pullback!)
    @eval function $pullback!(Δt::AbstractTensorMap, F, ΔF; kwargs...)
        foreachblock(Δt) do c, (b,)
            Fc = block.(F, Ref(c))
            ΔFc = block.(ΔF, Ref(c))
            return $pullback!(b, Fc, ΔFc; kwargs...)
        end
        return Δt
    end
end
