# DiagonalTensorMap
# -----------------
_repack_diagonal(d::DiagonalTensorMap) = Diagonal(d.data)

for f in (:svd_compact, :svd_full, :svd_trunc, :svd_vals, :qr_compact, :qr_full, :qr_null,
          :lq_compact, :lq_full, :lq_null, :eig_full, :eig_trunc, :eig_vals, :eigh_full,
          :eigh_trunc, :eigh_vals, :left_polar, :right_polar)
    @eval copy_input(::typeof($f), d::DiagonalTensorMap) = copy(d)
end

for f! in (:eig_full!, :eig_trunc!)
    @eval function initialize_output(::typeof($f!), d::AbstractTensorMap,
                                     ::DiagonalAlgorithm)
        return d, similar(d)
    end
end

for f! in (:eigh_full!, :eigh_trunc!)
    @eval function initialize_output(::typeof($f!), d::AbstractTensorMap,
                                     ::DiagonalAlgorithm)
        if scalartype(d) <: Real
            return d, similar(d)
        else
            return similar(d, real(scalartype(d))), similar(d)
        end
    end
end

for f! in (:qr_full!, :qr_compact!)
    @eval function initialize_output(::typeof($f!), d::AbstractTensorMap,
                                     ::DiagonalAlgorithm)
        return d, similar(d)
    end
    # to avoid ambiguities
    @eval function initialize_output(::typeof($f!), d::AdjointTensorMap,
                                     ::DiagonalAlgorithm)
        return d, similar(d)
    end
end
for f! in (:lq_full!, :lq_compact!)
    @eval function initialize_output(::typeof($f!), d::AbstractTensorMap,
                                     ::DiagonalAlgorithm)
        return similar(d), d
    end
    # to avoid ambiguities
    @eval function initialize_output(::typeof($f!), d::AdjointTensorMap,
                                     ::DiagonalAlgorithm)
        return similar(d), d
    end
end

function initialize_output(::typeof(left_orth!), d::DiagonalTensorMap)
    return d, similar(d)
end
function initialize_output(::typeof(right_orth!), d::DiagonalTensorMap)
    return similar(d), d
end

function initialize_output(::typeof(svd_full!), t::AbstractTensorMap, ::DiagonalAlgorithm)
    V_cod = fuse(codomain(t))
    V_dom = fuse(domain(t))
    U = similar(t, codomain(t) ← V_cod)
    S = DiagonalTensorMap{real(scalartype(t))}(undef, V_cod ← V_dom)
    Vᴴ = similar(t, V_dom ← domain(t))
    return U, S, Vᴴ
end

for f! in
    (:qr_full!, :qr_compact!, :lq_full!, :lq_compact!, :eig_full!, :eig_trunc!, :eigh_full!,
     :eigh_trunc!, :right_orth!, :left_orth!)
    @eval function $f!(d::DiagonalTensorMap, F, alg::DiagonalAlgorithm)
        check_input($f!, d, F, alg)
        $f!(_repack_diagonal(d), _repack_diagonal.(F), alg)
        return F
    end
end

for f! in (:qr_full!, :qr_compact!)
    @eval function check_input(::typeof($f!), d::AbstractTensorMap, QR,
                               ::DiagonalAlgorithm)
        Q, R = QR
        @assert d isa DiagonalTensorMap
        @assert Q isa DiagonalTensorMap && R isa DiagonalTensorMap
        @check_scalar Q d
        @check_scalar R d
        @check_space(Q, space(d))
        @check_space(R, space(d))

        return nothing
    end
end

for f! in (:lq_full!, :lq_compact!)
    @eval function check_input(::typeof($f!), d::AbstractTensorMap, LQ,
                               ::DiagonalAlgorithm)
        L, Q = LQ
        @assert d isa DiagonalTensorMap
        @assert Q isa DiagonalTensorMap && L isa DiagonalTensorMap
        @check_scalar Q d
        @check_scalar L d
        @check_space(Q, space(d))
        @check_space(L, space(d))

        return nothing
    end
end

# disambiguate
svd_compact!(t::AbstractTensorMap, USVᴴ, alg::DiagonalAlgorithm) = svd_full!(t, USVᴴ, alg)

# f_vals
# ------

for f! in (:eig_vals!, :eigh_vals!, :svd_vals!)
    @eval function $f!(d::AbstractTensorMap, V, alg::DiagonalAlgorithm)
        check_input($f!, d, V, alg)
        $f!(_repack_diagonal(d), diagview(_repack_diagonal(V)), alg)
        return V
    end
    @eval function initialize_output(::typeof($f!), d::DiagonalTensorMap,
                                     alg::DiagonalAlgorithm)
        data = initialize_output($f!, _repack_diagonal(d), alg)
        return DiagonalTensorMap(data, d.domain)
    end
end

function check_input(::typeof(eig_full!), t::DiagonalTensorMap, DV, ::DiagonalAlgorithm)
    domain(t) == codomain(t) ||
        throw(ArgumentError("Eigenvalue decomposition requires square input tensor"))

    D, V = DV

    @assert D isa DiagonalTensorMap
    @assert V isa AbstractTensorMap

    # scalartype checks
    @check_scalar D t
    @check_scalar V t

    # space checks
    @check_space D space(t)
    @check_space V space(t)

    return nothing
end

function check_input(::typeof(eigh_full!), t::DiagonalTensorMap, DV, ::DiagonalAlgorithm)
    domain(t) == codomain(t) ||
        throw(ArgumentError("Eigenvalue decomposition requires square input tensor"))

    D, V = DV

    @assert D isa DiagonalTensorMap
    @assert V isa AbstractTensorMap

    # scalartype checks
    @check_scalar D t real
    @check_scalar V t

    # space checks
    @check_space D space(t)
    @check_space V space(t)

    return nothing
end

function check_input(::typeof(eig_vals!), t::AbstractTensorMap, D, ::DiagonalAlgorithm)
    @assert D isa DiagonalTensorMap
    @check_scalar D t
    @check_space D space(t)
    return nothing
end

function check_input(::typeof(eigh_vals!), t::AbstractTensorMap, D, ::DiagonalAlgorithm)
    @assert D isa DiagonalTensorMap
    @check_scalar D t real
    @check_space D space(t)
    return nothing
end

function check_input(::typeof(svd_vals!), t::AbstractTensorMap, D, ::DiagonalAlgorithm)
    @assert D isa DiagonalTensorMap
    @check_scalar D t real
    @check_space D space(t)
    return nothing
end
