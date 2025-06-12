# Factorizations rules
# --------------------
function ChainRulesCore.rrule(::typeof(TensorKit.tsvd!), t::AbstractTensorMap;
                              trunc::TruncationStrategy=TensorKit.notrunc(),
                              kwargs...)
    # TODO: I think we can use tsvd! here without issues because we don't actually require
    # the data of `t` anymore.
    USVᴴ = tsvd(t; trunc=TensorKit.notrunc(), alg)

    if trunc != TensorKit.notrunc() && !isempty(blocksectors(t))
        USVᴴ′ = MatrixAlgebraKit.truncate!(svd_trunc!, USVᴴ, trunc)
    else
        USVᴴ′ = USVᴴ
    end

    function tsvd!_pullback(ΔUSVᴴ′)
        ΔUSVᴴ = unthunk.(ΔUSVᴴ′)
        Δt = similar(t)
        foreachblock(Δt) do (c, b)
            USVᴴc = block.(USVᴴ, Ref(c))
            ΔUSVᴴc = block.(ΔUSVᴴ, Ref(c))
            svd_compact_pullback!(b, USVᴴc, ΔUSVᴴc)
            return nothing
        end
        return NoTangent(), Δt
    end
    tsvd!_pullback(::NTuple{3,ZeroTangent}) = NoTangent(), ZeroTangent()

    return USVᴴ′, tsvd!_pullback
end

function ChainRulesCore.rrule(::typeof(LinearAlgebra.svdvals!), t::AbstractTensorMap)
    U, S, V⁺ = tsvd(t)
    s = diag(S)
    project_t = ProjectTo(t)

    function svdvals_pullback(Δs′)
        Δs = unthunk(Δs′)
        ΔS = diagm(codomain(S), domain(S), Δs)
        return NoTangent(), project_t(U * ΔS * V⁺)
    end

    return s, svdvals_pullback
end

function ChainRulesCore.rrule(::typeof(TensorKit.eig!), t::AbstractTensorMap; kwargs...)
    DV = eig(t; kwargs...)

    function eig!_pullback(ΔDV′)
        ΔDV = unthunk.(ΔDV′)
        Δt = similar(t)
        foreachblock(Δt) do (c, b)
            DVc = block.(DV, Ref(c))
            ΔDVc = block.(ΔDV, Ref(c))
            eig_full_pullback!(b, DVc, ΔDVc)
            return nothing
        end
        return NoTangent(), Δt
    end
    eig!_pullback(::NTuple{2,ZeroTangent}) = NoTangent(), ZeroTangent()

    return DV, eig!_pullback
end

function ChainRulesCore.rrule(::typeof(TensorKit.eigh!), t::AbstractTensorMap; kwargs...)
    DV = eigh(t; kwargs...)

    function eigh!_pullback(ΔDV′)
        ΔDV = unthunk.(ΔDV′)
        Δt = similar(t)
        foreachblock(Δt) do (c, b)
            DVc = block.(DV, Ref(c))
            ΔDVc = block.(ΔDV, Ref(c))
            eigh_full_pullback!(b, DVc, ΔDVc)
            return nothing
        end
        return NoTangent(), Δt
    end
    eigh!_pullback(::NTuple{2,ZeroTangent}) = NoTangent(), ZeroTangent()

    return DV, eigh!_pullback
end

function ChainRulesCore.rrule(::typeof(LinearAlgebra.eigvals!), t::AbstractTensorMap;
                              sortby=nothing, kwargs...)
    @assert sortby === nothing "only `sortby=nothing` is supported"
    (D, _), eig_pullback = rrule(TensorKit.eig!, t; kwargs...)
    d = diag(D)
    project_t = ProjectTo(t)
    function eigvals_pullback(Δd′)
        Δd = unthunk(Δd′)
        ΔD = diagm(codomain(D), domain(D), Δd)
        return NoTangent(), project_t(eig_pullback((ΔD, ZeroTangent()))[2])
    end

    return d, eigvals_pullback
end

function ChainRulesCore.rrule(::typeof(leftorth!), t::AbstractTensorMap; alg=QRpos())
    alg isa TensorKit.QR || alg isa TensorKit.QRpos ||
        error("only `alg=QR()` and `alg=QRpos()` are supported")
    Q, R = leftorth(t; alg)
    function leftorth!_pullback((_ΔQ, _ΔR))
        ΔQ, ΔR = unthunk(_ΔQ), unthunk(_ΔR)
        Δt = similar(t)
        for (c, b) in blocks(Δt)
            qr_pullback!(b, block(Q, c), block(R, c), block(ΔQ, c), block(ΔR, c))
        end
        return NoTangent(), Δt
    end
    leftorth!_pullback(::Tuple{ZeroTangent,ZeroTangent}) = NoTangent(), ZeroTangent()
    return (Q, R), leftorth!_pullback
end

function ChainRulesCore.rrule(::typeof(rightorth!), t::AbstractTensorMap; alg=LQpos())
    alg isa TensorKit.LQ || alg isa TensorKit.LQpos ||
        error("only `alg=LQ()` and `alg=LQpos()` are supported")
    L, Q = rightorth(t; alg)
    function rightorth!_pullback((_ΔL, _ΔQ))
        ΔL, ΔQ = unthunk(_ΔL), unthunk(_ΔQ)
        Δt = similar(t)
        for (c, b) in blocks(Δt)
            lq_pullback!(b, block(L, c), block(Q, c), block(ΔL, c), block(ΔQ, c))
        end
        return NoTangent(), Δt
    end
    rightorth!_pullback(::Tuple{ZeroTangent,ZeroTangent}) = NoTangent(), ZeroTangent()
    return (L, Q), rightorth!_pullback
end

# Corresponding matrix factorisations: implemented as mutating methods
# ---------------------------------------------------------------------
# helper routines
safe_inv(a, tol) = abs(a) < tol ? zero(a) : inv(a)

function lowertriangularind(A::AbstractMatrix)
    m, n = size(A)
    I = Vector{Int}(undef, div(m * (m - 1), 2) + m * (n - m))
    offset = 0
    for j in 1:n
        r = (j + 1):m
        I[offset .- j .+ r] = (j - 1) * m .+ r
        offset += length(r)
    end
    return I
end

function uppertriangularind(A::AbstractMatrix)
    m, n = size(A)
    I = Vector{Int}(undef, div(m * (m - 1), 2) + m * (n - m))
    offset = 0
    for i in 1:m
        r = (i + 1):n
        I[offset .- i .+ r] = i .+ m .* (r .- 1)
        offset += length(r)
    end
    return I
end

function qr_pullback!(ΔA::AbstractMatrix, Q::AbstractMatrix, R::AbstractMatrix, ΔQ, ΔR;
                      tol::Real=default_pullback_gaugetol(R))
    Rd = view(R, diagind(R))
    p = something(findlast(≥(tol) ∘ abs, Rd), 0)
    m, n = size(R)

    Q1 = view(Q, :, 1:p)
    R1 = view(R, 1:p, :)
    R11 = view(R, 1:p, 1:p)

    ΔA1 = view(ΔA, :, 1:p)
    ΔQ1 = view(ΔQ, :, 1:p)
    ΔR1 = view(ΔR, 1:p, :)

    M = similar(R, (p, p))
    ΔR isa AbstractZero || mul!(M, ΔR1, R1')
    ΔQ isa AbstractZero || mul!(M, Q1', ΔQ1, -1, !(ΔR isa AbstractZero))
    view(M, lowertriangularind(M)) .= conj.(view(M, uppertriangularind(M)))
    if eltype(M) <: Complex
        Md = view(M, diagind(M))
        Md .= real.(Md)
    end

    ΔA1 .= ΔQ1
    mul!(ΔA1, Q1, M, +1, 1)

    if n > p
        R12 = view(R, 1:p, (p + 1):n)
        ΔA2 = view(ΔA, :, (p + 1):n)
        ΔR12 = view(ΔR, 1:p, (p + 1):n)

        if ΔR isa AbstractZero
            ΔA2 .= zero(eltype(ΔA))
        else
            mul!(ΔA2, Q1, ΔR12)
            mul!(ΔA1, ΔA2, R12', -1, 1)
        end
    end
    if m > p && !(ΔQ isa AbstractZero) # case where R is not full rank
        Q2 = view(Q, :, (p + 1):m)
        ΔQ2 = view(ΔQ, :, (p + 1):m)
        Q1dΔQ2 = Q1' * ΔQ2
        Δgauge = norm(mul!(copy(ΔQ2), Q1, Q1dΔQ2, -1, 1), Inf)
        Δgauge < tol ||
            @warn "`qr` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
        mul!(ΔA1, Q2, Q1dΔQ2', -1, 1)
    end
    rdiv!(ΔA1, UpperTriangular(R11)')
    return ΔA
end

function lq_pullback!(ΔA::AbstractMatrix, L::AbstractMatrix, Q::AbstractMatrix, ΔL, ΔQ;
                      tol::Real=default_pullback_gaugetol(L))
    Ld = view(L, diagind(L))
    p = something(findlast(≥(tol) ∘ abs, Ld), 0)
    m, n = size(L)

    L1 = view(L, :, 1:p)
    L11 = view(L, 1:p, 1:p)
    Q1 = view(Q, 1:p, :)

    ΔA1 = view(ΔA, 1:p, :)
    ΔQ1 = view(ΔQ, 1:p, :)
    ΔL1 = view(ΔL, :, 1:p)

    M = similar(L, (p, p))
    ΔL isa AbstractZero || mul!(M, L1', ΔL1)
    ΔQ isa AbstractZero || mul!(M, ΔQ1, Q1', -1, !(ΔL isa AbstractZero))
    view(M, uppertriangularind(M)) .= conj.(view(M, lowertriangularind(M)))
    if eltype(M) <: Complex
        Md = view(M, diagind(M))
        Md .= real.(Md)
    end

    ΔA1 .= ΔQ1
    mul!(ΔA1, M, Q1, +1, 1)

    if m > p
        L21 = view(L, (p + 1):m, 1:p)
        ΔA2 = view(ΔA, (p + 1):m, :)
        ΔL21 = view(ΔL, (p + 1):m, 1:p)

        if ΔL isa AbstractZero
            ΔA2 .= zero(eltype(ΔA))
        else
            mul!(ΔA2, ΔL21, Q1)
            mul!(ΔA1, L21', ΔA2, -1, 1)
        end
    end
    if n > p && !(ΔQ isa AbstractZero) # case where R is not full rank
        Q2 = view(Q, (p + 1):n, :)
        ΔQ2 = view(ΔQ, (p + 1):n, :)
        ΔQ2Q1d = ΔQ2 * Q1'
        Δgauge = norm(mul!(copy(ΔQ2), ΔQ2Q1d, Q1, -1, 1))
        Δgauge < tol ||
            @warn "`lq` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
        mul!(ΔA1, ΔQ2Q1d', Q2, -1, 1)
    end
    ldiv!(LowerTriangular(L11)', ΔA1)
    return ΔA
end

function default_pullback_gaugetol(a)
    n = norm(a, Inf)
    return eps(eltype(n))^(3 / 4) * max(n, one(n))
end
