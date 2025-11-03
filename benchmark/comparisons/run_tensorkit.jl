include("bonds.jl")
include("tensorkit_timers.jl")

using LinearAlgebra: LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

using TensorOperations: TensorOperations
TensorOperations.enable_cache(; maxrelsize = 0.7)
using TensorKit: TensorKit
using TensorKit: ℂ, Z2Space, U1Space
using DelimitedFiles

K = 100 # number of repititions; scale up for more reliable testing.

# MPO Contractions

# TRIVIAL = no symmetry
N = length(mpodims_triv[1]) # number of different cases
mpo_triv_times = zeros(Float64, (K, N))
mpo_triv_times_gc = zeros(Float64, (K, N))
for i in 1:N
    Dmps = mpodims_triv[1][i]
    Dmpo = mpodims_triv[2][i]
    Dphys = mpodims_triv[3][i]

    Vmps = ℂ^Dmps
    Vmpo = ℂ^Dmpo
    Vphys = ℂ^Dphys

    mpo_timer = TensorKitTimers.mpo_timer(; Vmps = Vmps, Vmpo = Vmpo, Vphys = Vphys)

    times, times_gc = mpo_timer(; outer = K)
    mpo_triv_times[:, i] = times
    mpo_triv_times_gc[:, i] = times_gc
    empty!(TensorOperations.cache)

    tavg = sum(times) / K

    println("MPO Contraction, trivial symmetry: case $i out of $N finished at average time $tavg")
end

writedlm("results/tensorkit_mpo_triv.txt", mpo_triv_times)
writedlm("results/tensorkit_mpo_triv_gc.txt", mpo_triv_times_gc)

# Z2 Symmetry
N = length(mpodims_z2[1]) # number of different cases
mpo_z2_times = zeros(Float64, (K, N))
mpo_z2_times_gc = zeros(Float64, (K, N))
for i in 1:N
    Dmps = distributeZ2(mpodims_z2[1][i])
    Dmpo = distributeZ2(mpodims_z2[2][i])
    Dphys = distributeZ2(mpodims_z2[3][i])

    Vmps = Z2Space((a => b for (a, b) in Dmps)...)
    Vmpo = Z2Space((a => b for (a, b) in Dmpo)...)
    Vphys = Z2Space((a => b for (a, b) in Dphys)...)

    mpo_timer = TensorKitTimers.mpo_timer(; Vmps = Vmps, Vmpo = Vmpo, Vphys = Vphys)

    times, times_gc = mpo_timer(; outer = K)
    mpo_z2_times[:, i] = times
    mpo_z2_times_gc[:, i] = times_gc
    empty!(TensorOperations.cache)

    tavg = sum(times) / K

    println("MPO Contraction, Z2 symmetry: case $i out of $N finished at average time $tavg")
end

writedlm("results/tensorkit_mpo_z2.txt", mpo_z2_times)
writedlm("results/tensorkit_mpo_z2_gc.txt", mpo_z2_times_gc)

# U1 Symmetry
N = length(mpodims_u1[1]) # number of different cases
mpo_u1_times = zeros(Float64, (K, N))
mpo_u1_times_gc = zeros(Float64, (K, N))
for i in 1:N
    Dmps = distributeU1(mpodims_u1[1][i])
    Dmpo = distributeU1(mpodims_u1[2][i])
    Dphys = distributeU1(mpodims_u1[3][i])

    Vmps = U1Space((a => b for (a, b) in Dmps)...)
    Vmpo = U1Space((a => b for (a, b) in Dmpo)...)
    Vphys = U1Space((a => b for (a, b) in Dphys)...)

    mpo_timer = TensorKitTimers.mpo_timer(; Vmps = Vmps, Vmpo = Vmpo, Vphys = Vphys)

    times, times_gc = mpo_timer(; outer = K)
    mpo_u1_times[:, i] = times
    mpo_u1_times_gc[:, i] = times_gc
    empty!(TensorOperations.cache)

    tavg = sum(times) / K

    println("MPO Contraction, U1 symmetry: case $i out of $N finished at average time $tavg")
end

writedlm("results/tensorkit_mpo_u1.txt", mpo_u1_times)
writedlm("results/tensorkit_mpo_u1_gc.txt", mpo_u1_times_gc)

## PEPO Contractions

# TRIVIAL = no symmetry
N = length(pepodims_triv[1]) # number of different cases
pepo_triv_times = zeros(Float64, (K, N))
pepo_triv_times_gc = zeros(Float64, (K, N))
for i in 1:N
    Dpeps = pepodims_triv[1][i]
    Dpepo = pepodims_triv[2][i]
    Dphys = pepodims_triv[3][i]
    Denv = pepodims_triv[4][i]

    Vpeps = ℂ^Dpeps
    Vpepo = ℂ^Dpepo
    Vphys = ℂ^Dphys
    Venv = ℂ^Denv

    pepo_timer = TensorKitTimers.pepo_timer(; Vpeps, Vpepo, Vphys, Venv)

    times, times_gc = pepo_timer(; outer = K)
    pepo_triv_times[:, i] = times
    pepo_triv_times_gc[:, i] = times_gc
    empty!(TensorOperations.cache)

    tavg = sum(times) / K

    println("PEPO Contraction, trivial symmetry: case $i out of $N finished at average time $tavg")
end

writedlm("results/tensorkit_pepo_triv.txt", pepo_triv_times)
writedlm("results/tensorkit_pepo_triv_gc.txt", pepo_triv_times_gc)

# Z2 Symmetry
N = length(pepodims_z2[1]) # number of different cases
pepo_z2_times = zeros(Float64, (K, N))
pepo_z2_times_gc = zeros(Float64, (K, N))
for i in 1:N
    Dpeps = distributeZ2(pepodims_z2[1][i])
    Dpepo = distributeZ2(pepodims_z2[2][i])
    Dphys = distributeZ2(pepodims_z2[3][i])
    Denv = distributeZ2(pepodims_z2[4][i])

    Vpeps = Z2Space((a => b for (a, b) in Dpeps)...)
    Vpepo = Z2Space((a => b for (a, b) in Dpepo)...)
    Vphys = Z2Space((a => b for (a, b) in Dphys)...)
    Venv = Z2Space((a => b for (a, b) in Denv)...)

    pepo_timer = TensorKitTimers.pepo_timer(; Vpeps, Vpepo, Vphys, Venv)

    times, times_gc = pepo_timer(; outer = K)
    pepo_z2_times[:, i] = times
    pepo_z2_times_gc[:, i] = times_gc
    empty!(TensorOperations.cache)

    tavg = sum(times) / K

    println("PEPO Contraction, Z2 symmetry: case $i out of $N finished at average time $tavg")
end

writedlm("results/tensorkit_pepo_z2.txt", pepo_z2_times)
writedlm("results/tensorkit_pepo_z2_gc.txt", pepo_z2_times_gc)

# U1 Symmetry
N = length(pepodims_u1[1]) # number of different cases
pepo_u1_times = zeros(Float64, (K, N))
pepo_u1_times_gc = zeros(Float64, (K, N))
for i in 1:N
    Dpeps = distributeU1(pepodims_u1[1][i])
    Dpepo = distributeU1(pepodims_u1[2][i])
    Dphys = distributeU1(pepodims_u1[3][i])
    Denv = distributeU1(pepodims_u1[4][i])

    Vpeps = U1Space((a => b for (a, b) in Dpeps)...)
    Vpepo = U1Space((a => b for (a, b) in Dpepo)...)
    Vphys = U1Space((a => b for (a, b) in Dphys)...)
    Venv = U1Space((a => b for (a, b) in Denv)...)

    pepo_timer = TensorKitTimers.pepo_timer(; Vpeps, Vpepo, Vphys, Venv)

    times, times_gc = pepo_timer(; outer = K)
    pepo_u1_times[:, i] = times
    pepo_u1_times_gc[:, i] = times_gc
    empty!(TensorOperations.cache)

    tavg = sum(times) / K

    println("PEPO Contraction, U1 symmetry: case $i out of $N finished at average time $tavg")
end

writedlm("results/tensorkit_pepo_u1.txt", pepo_u1_times)
writedlm("results/tensorkit_pepo_u1_gc.txt", pepo_u1_times_gc)

# ## MERA Contractions

# TRIVIAL = no symmetry
N = length(meradims_triv[1]) # number of different cases
mera_triv_times = zeros(Float64, (K, N))
mera_triv_times_gc = zeros(Float64, (K, N))
for i in 1:N
    Dmera = meradims_triv[1][i]

    Vmera = ℂ^Dmera

    mera_timer = TensorKitTimers.mera_timer(; Vmera = Vmera)

    times, times_gc = mera_timer(; outer = K)

    mera_triv_times[:, i] = times
    mera_triv_times_gc[:, i] = times_gc
    empty!(TensorOperations.cache)

    tavg = sum(times) / K

    println("MERA Contraction, trivial symmetry: case $i out of $N finished at average time $tavg")
end

writedlm("results/tensorkit_mera_triv.txt", mera_triv_times)
writedlm("results/tensorkit_mera_triv_gc.txt", mera_triv_times_gc)

# Z2 Symmetry
N = length(meradims_z2[1]) # number of different cases
mera_z2_times = zeros(Float64, (K, N))
mera_z2_times_gc = zeros(Float64, (K, N))
for i in 1:N
    Dmera = distributeZ2(meradims_z2[1][i])

    Vmera = Z2Space((a => b for (a, b) in Dmera)...)

    mera_timer = TensorKitTimers.mera_timer(; Vmera = Vmera)

    times, times_gc = mera_timer(; outer = K)

    mera_z2_times[:, i] = times
    mera_z2_times_gc[:, i] = times_gc
    empty!(TensorOperations.cache)

    tavg = sum(times) / K

    println("MERA Contraction, Z2 symmetry: case $i out of $N finished at average time $tavg")
end

writedlm("results/tensorkit_mera_z2.txt", mera_z2_times)
writedlm("results/tensorkit_mera_z2_gc.txt", mera_z2_times_gc)

# U1 Symmetry
N = length(meradims_u1[1]) # number of different cases
mera_u1_times = zeros(Float64, (K, N))
mera_u1_times_gc = zeros(Float64, (K, N))
for i in 1:N
    Dmera = distributeU1(meradims_u1[1][i])

    Vmera = U1Space((a => b for (a, b) in Dmera)...)

    mera_timer = TensorKitTimers.mera_timer(; Vmera = Vmera)

    times, times_gc = mera_timer(; outer = K)

    mera_u1_times[:, i] = times
    mera_u1_times_gc[:, i] = times_gc
    empty!(TensorOperations.cache)

    tavg = sum(times) / K

    println("MERA Contraction, U1 symmetry: case $i out of $N finished at average time $tavg")
end

writedlm("results/tensorkit_mera_u1.txt", mera_u1_times)
writedlm("results/tensorkit_mera_u1_gc.txt", mera_u1_times_gc)
