using DelimitedFiles
using Plots, StatsPlots, Measures
gr()

M = 1 # markersize

include("bonds.jl")
# MPO results

numplots = 8
mpoplots = Matrix{Any}(undef, (numplots, 3))

data = ["_mpo_triv.txt", "_mpo_z2.txt", "_mpo_u1.txt"]
dims = [mpodims_triv, mpodims_z2, mpodims_u1]
name = ["Trivial", "Z2 symmetry", "U1 symmetry"]
for j in 1:3
    s = data[j]
    data_tensorkit = try
        readdlm("results/tensorkit" * s)
    catch
        nothing
    end
    ymin = minimum(data_tensorkit)
    ymax = maximum(data_tensorkit)
    data_itensors = try
        readdlm("results/itensors" * s)
    catch
        nothing
    end
    if !isnothing(data_itensors)
        ymin = min(ymin, minimum(data_itensors))
        ymax = max(ymax, maximum(data_itensors))
    end
    data_tenpy = try
        readdlm("results/tenpy" * s)
    catch
        nothing
    end
    if !isnothing(data_tenpy)
        ymin = min(ymin, minimum(data_tenpy))
        ymax = max(ymax, maximum(data_tenpy))
    end
    data_tensornetwork = try
        readdlm("results/tensornetwork" * s)
    catch
        nothing
    end
    if !isnothing(data_tensornetwork)
        ymin = min(ymin, minimum(data_tensornetwork))
        ymax = max(ymax, maximum(data_tensornetwork))
    end
    ymin = 10^(floor(log10(ymin)))
    ymax = 10^(ceil(log10(ymax)))

    for i in 1:numplots
        Dmps = dims[j][1][i]
        Dmpo = dims[j][2][i]
        Dphys = dims[j][3][i]

        p = boxplot(data_tensorkit[:, i]; label = "TensorKit.jl", markersize = M)
        !isnothing(data_itensors) &&
            boxplot!(p, data_itensors[:, i]; label = "ITensors.jl", markersize = M)
        !isnothing(data_tenpy) &&
            boxplot!(p, data_tenpy[:, i]; label = "Tenpy", markersize = M)
        !isnothing(data_tensornetwork) &&
            boxplot!(p, data_tensornetwork[:, i]; label = "TensorNetwork", markersize = M)
        plot!(
            p; yscale = :log10, yminorticks = true, xticks = [], xgrid = false, xshowaxis = false,
            ylim = (ymin, ymax)
        )
        if i == 1
            plot!(p; title = "(D, M, d) = ($Dmps, $Dmpo, $Dphys)", titlefontsize = 10)
            plot!(
                p; yguide = name[j], yguide_position = :left, yguidefontsize = 10,
                ytickfontsize = 8, left_margin = 6mm
            )
        else
            plot!(p; title = "($Dmps, $Dmpo, $Dphys)", titlefontsize = 10)
            plot!(p; left_margin = -5mm, yformatter = _ -> "", yshowaxis = false)
        end
        if i == numplots && j == 3
            plot!(p; legendfontsize = 10, legend = :bottomright)
        else
            plot!(p; legend = false)
        end
        mpoplots[i, j] = p
    end
end
plotgrid = plot(
    mpoplots...; layout = grid(3, numplots), link = :y, thickness_scaling = 1, size = (1500, 800)
)
savefig(plotgrid, "mporesults.pdf")
