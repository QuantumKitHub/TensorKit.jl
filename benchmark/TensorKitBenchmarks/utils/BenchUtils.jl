module BenchUtils

export parse_type, expand_kwargs
export generate_space

using TensorKit

parse_type(x::String) = eval(Meta.parse(x))

function expand_kwargs(params::Dict)
    const_params = NamedTuple(
        Symbol(key) => value for (key, value) in params if !(value isa Vector)
    )
    nonconst_keys = Tuple(Symbol(key) for (key, value) in params if value isa Vector)
    nonconst_vals = (value for value in values(params) if value isa Vector)
    return Iterators.map(Iterators.product(nonconst_vals...)) do expanded_vals
        return merge(const_params, NamedTuple{nonconst_keys}(expanded_vals))
    end
end

"""
    generate_space(::Type{I}, D::Int; sigma::Real=1.0)

Creates a (graded) vector space with sectortype `I` and total dimension `D`, where the distribution of charges is controlled through a spread parameter `sigma`.
"""
function generate_space(::Type{Trivial}, D::Int, sigma::Nothing = nothing)
    return ComplexSpace(round(Int, D))
end
function generate_space(::Type{Z2Irrep}, D::Int, sigma::Real = 0.5)
    D_even = ceil(Int, sigma * D)
    D_odd = D - D_even
    return Z2Space(0 => D_even, 1 => D_odd)
end
function generate_space(::Type{U1Irrep}, D::Int, sigma::Real = 0.5)
    # use ceil here to avoid getting stuck
    normal_pdf = let D = D
        x -> ceil(Int, D * exp(-0.5 * (x / sigma)^2) / (sigma * sqrt(2π)))
    end

    sectors = U1Irrep[]
    dims = Int[]

    for sector in values(U1Irrep)
        round(sector.charge) == sector.charge || continue
        d = normal_pdf(sector.charge)
        push!(sectors, sector)
        push!(dims, d)
        D -= d
        D < 1 && break

        if abs(sector.charge) > 20
            error()
        end
    end

    return U1Space((s => d for (s, d) in zip(sectors, dims))...)
end
function generate_space(::Type{SU2Irrep}, D::Int, sigma::Real = 0.5)
    normal_pdf = let D = D
        x -> D * exp(-0.5 * (x / sigma)^2) / (sigma * sqrt(2π))
    end

    sectors = SU2Irrep[]
    dims = Int[]

    for sector in values(SU2Irrep)
        d = ceil(Int, normal_pdf(sector.j) / dim(sector))
        push!(sectors, sector)
        push!(dims, d)
        D -= d * dim(sector)
        D < 1 && break
    end

    return SU2Space((s => d for (s, d) in zip(sectors, dims))...)
end

end
