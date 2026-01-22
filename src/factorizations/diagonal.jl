# DiagonalTensorMap
# -----------------
_repack_diagonal(d::DiagonalTensorMap) = Diagonal(d.data)
_repack_diagonal(d::SectorVector) = Diagonal(parent(d))

MAK.diagview(t::DiagonalTensorMap) = SectorVector(t.data, TensorKit.diagonalblockstructure(space(t)))

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
        return similar(d, complex(scalartype(d))), similar(d, complex(scalartype(d)), space(d))
    end
end

for f! in (:eigh_full!, :eigh_trunc!)
    @eval function MAK.initialize_output(
            ::typeof($f!), d::AbstractTensorMap, ::DiagonalAlgorithm
        )
        if scalartype(d) <: Real
            return d, similar(d, space(d))
        else
            return similar(d, real(scalartype(d))), similar(d, space(d))
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
    S = similar_diagonal(t, real(scalartype(t)), V_cod)
    Vᴴ = similar(t, V_dom ← domain(t))
    return U, S, Vᴴ
end

for f! in (:qr_full!, :qr_compact!, :lq_full!, :lq_compact!, :right_orth!, :left_orth!)
    @eval function MAK.$f!(d::DiagonalTensorMap, F, alg::DiagonalAlgorithm)
        $f!(_repack_diagonal(d), _repack_diagonal.(F), alg)
        return F
    end
end

# disambiguate
function MAK.svd_compact!(t::AbstractTensorMap, USVᴴ, alg::DiagonalAlgorithm)
    return svd_full!(t, USVᴴ, alg)
end

# For diagonal inputs we don't have to promote the scalartype since we know they are symmetric
function MAK.initialize_output(::typeof(eig_vals!), t::AbstractTensorMap, alg::DiagonalAlgorithm)
    V_D = fuse(domain(t))
    Tc = complex(scalartype(t))
    A = similarstoragetype(t, Tc)
    return SectorVector{Tc, sectortype(t), A}(undef, V_D)
end
