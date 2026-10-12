# DiagonalTensorMap
# -----------------
_repack_diagonal(d::DiagonalTensorMap) = Diagonal(d.data)
_repack_diagonal(d::SectorVector) = Diagonal(parent(d))

MAK.diagonal(t::SectorVector) = DiagonalTensorMap(t)
MAK.diagview(t::DiagonalTensorMap) = SectorVector(t.data, TensorKit.diagonalblockstructure(space(t)))

for f in (
        :svd_compact, :svd_full, :svd_trunc, :svd_vals, :qr_compact, :qr_full, :qr_null,
        :lq_compact, :lq_full, :lq_null, :eig_full, :eig_trunc, :eig_vals, :eigh_full,
        :eigh_trunc, :eigh_vals, :left_polar, :right_polar,
    )
    @eval MAK.copy_input(::typeof($f), d::DiagonalTensorMap) = copy(d)
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

for (f!, bf!) in ((:svd_vals!, :batched_svd_vals!),)
    # diagonals don't support batching yet
    @eval function MAK.$f!(t::AbstractTensorMap, N, alg::DiagonalAlgorithm)
        @timeit_debug GLOBAL_TIMER $(string(f!)) begin
            foreachblock(t, N) do _, (tblock, Nblock)
                @timeit_debug GLOBAL_TIMER "dense: MatrixAlgebraKit" begin
                    Nblock′ = $f!(tblock, Nblock, alg)
                    # deal with the case where the output is not the same as the input
                    Nblock === Nblock′ || copy!(Nblock, Nblock′)
                end
                return nothing
            end
        end
        return N
    end
end

for (f!, bf!) in ((:svd_full!, :batched_svd_full!),)
    @eval function MAK.$f!(t::AbstractTensorMap, F, alg::DiagonalAlgorithm)
        @timeit_debug GLOBAL_TIMER $(string(f!)) begin
            U, S, Vᴴ = F
            foreachblock(t, U, S, Vᴴ) do _, (tblock, Fblocks...)
                @timeit_debug GLOBAL_TIMER "dense: MatrixAlgebraKit" begin
                    Fblocks′ = $f!(tblock, Fblocks, alg)
                    # deal with the case where the output is not in-place
                    for (b′, b) in zip(Fblocks′, Fblocks)
                        b === b′ || copy!(b, b′)
                    end
                end
                return nothing
            end
            return F
        end
    end
end

