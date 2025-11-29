# DiagonalTensorMap
# -----------------
_repack_diagonal(d::DiagonalTensorMap) = Diagonal(d.data)

for f in (
        :svd_compact, :svd_full, :svd_trunc, :svd_vals, :qr_compact, :qr_full, :qr_null,
        :lq_compact, :lq_full, :lq_null, :eig_full, :eig_trunc, :eig_vals, :eigh_full,
        :eigh_trunc, :eigh_vals, :left_polar, :right_polar,
    )
    @eval MAK.copy_input(::typeof($f), d::DiagonalTensorMap) = copy(d)
end

for f! in (:eig_full!, :eig_trunc!)
    @eval function MAK.initialize_output(
            ::typeof($f!), d::AbstractTensorMap, ::DiagonalAlgorithm
        )
        return d, similar(d)
    end
end

for f! in (:eigh_full!, :eigh_trunc!)
    @eval function MAK.initialize_output(
            ::typeof($f!), d::AbstractTensorMap, ::DiagonalAlgorithm
        )
        if scalartype(d) <: Real
            return d, similar(d)
        else
            return similar(d, real(scalartype(d))), similar(d)
        end
    end
end

for f! in (:qr_full!, :qr_compact!)
    @eval function MAK.initialize_output(
            ::typeof($f!), d::AbstractTensorMap, ::DiagonalAlgorithm
        )
        return d, similar(d)
    end
    # to avoid ambiguities
    @eval function MAK.initialize_output(
            ::typeof($f!), d::AdjointTensorMap, ::DiagonalAlgorithm
        )
        return d, similar(d)
    end
end
for f! in (:lq_full!, :lq_compact!)
    @eval function MAK.initialize_output(
            ::typeof($f!), d::AbstractTensorMap, ::DiagonalAlgorithm
        )
        return similar(d), d
    end
    # to avoid ambiguities
    @eval function MAK.initialize_output(
            ::typeof($f!), d::AdjointTensorMap, ::DiagonalAlgorithm
        )
        return similar(d), d
    end
end

function MAK.initialize_output(::typeof(left_orth!), d::DiagonalTensorMap)
    return d, similar(d)
end
function MAK.initialize_output(::typeof(right_orth!), d::DiagonalTensorMap)
    return similar(d), d
end

function MAK.initialize_output(
        ::typeof(svd_full!), t::AbstractTensorMap, ::DiagonalAlgorithm
    )
    V_cod = fuse(codomain(t))
    V_dom = fuse(domain(t))
    U = similar(t, codomain(t) ← V_cod)
    S = DiagonalTensorMap{real(scalartype(t))}(undef, V_cod ← V_dom)
    Vᴴ = similar(t, V_dom ← domain(t))
    return U, S, Vᴴ
end

for f! in
    (
        :qr_full!, :qr_compact!, :lq_full!, :lq_compact!, :eig_full!, :eig_trunc!, :eigh_full!,
        :eigh_trunc!, :right_orth!, :left_orth!,
    )
    @eval function MAK.$f!(d::DiagonalTensorMap, F, alg::DiagonalAlgorithm)
        $f!(_repack_diagonal(d), _repack_diagonal.(F), alg)
        return F
    end
end

# disambiguate
function MAK.svd_compact!(t::AbstractTensorMap, USVᴴ, alg::DiagonalAlgorithm)
    return svd_full!(t, USVᴴ, alg)
end

# f_vals
# ------
for f! in (:eig_vals!, :eigh_vals!, :svd_vals!)
    @eval function MAK.$f!(d::AbstractTensorMap, V, alg::DiagonalAlgorithm)
        $f!(_repack_diagonal(d), diagview(_repack_diagonal(V)), alg)
        return V
    end
    @eval function MAK.initialize_output(
            ::typeof($f!), d::DiagonalTensorMap, alg::DiagonalAlgorithm
        )
        data = MAK.initialize_output($f!, _repack_diagonal(d), alg)
        return DiagonalTensorMap(data, d.domain)
    end
end
