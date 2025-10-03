include("benchtools.jl")
module TensorKitTimers
    using TensorKit
    using ..Timers: Timers

    function mpo_timer(f = randn, T = Float64; Vmpo, Vmps, Vphys)
        A = Tensor(f, T, Vmps ⊗ Vphys ⊗ Vmps')
        M = Tensor(f, T, Vmpo ⊗ Vphys ⊗ Vphys' ⊗ Vmpo')
        FL = Tensor(f, T, Vmps ⊗ Vmpo' ⊗ Vmps')
        FR = Tensor(f, T, Vmps ⊗ Vmpo ⊗ Vmps')

        return Timers.Timer(A, M, FL, FR) do A, M, FL, FR
            @tensor C = FL[4, 2, 1] * A[1, 3, 6] * M[2, 5, 3, 7] * conj(A[4, 5, 8]) *
                FR[6, 7, 8]
            return C
        end
    end

    function pepo_timer(f = randn, T = Float64; Vpepo, Vpeps, Venv, Vphys)
        A = Tensor(f, T, Vpeps ⊗ Vpeps ⊗ Vphys ⊗ Vpeps' ⊗ Vpeps')
        P = Tensor(f, T, Vpepo ⊗ Vpepo ⊗ Vphys ⊗ Vphys' ⊗ Vpepo' ⊗ Vpepo')
        FL = Tensor(f, T, Venv ⊗ Vpeps ⊗ Vpepo' ⊗ Vpeps' ⊗ Venv')
        FD = Tensor(f, T, Venv ⊗ Vpeps ⊗ Vpepo' ⊗ Vpeps' ⊗ Venv')
        FR = Tensor(f, T, Venv ⊗ Vpeps ⊗ Vpepo ⊗ Vpeps' ⊗ Venv')
        FU = Tensor(f, T, Venv ⊗ Vpeps ⊗ Vpepo ⊗ Vpeps' ⊗ Venv')
        return Timers.Timer(A, P, FL, FD, FR, FU) do A, P, FL, FD, FR, FU
            @tensor C = FL[18, 7, 4, 2, 1] * FU[1, 3, 6, 9, 10] *
                A[2, 17, 5, 3, 11] * P[4, 16, 8, 5, 6, 12] * conj(A[7, 15, 8, 9, 13]) *
                FR[10, 11, 12, 13, 14] * FD[14, 15, 16, 17, 18]
            return C
        end
    end

    function mera_timer(f = randn, T = Float64; Vmera)
        u = Tensor(f, T, Vmera ⊗ Vmera ⊗ Vmera' ⊗ Vmera')
        w = Tensor(f, T, Vmera ⊗ Vmera ⊗ Vmera')
        ρ = Tensor(f, T, Vmera ⊗ Vmera ⊗ Vmera ⊗ Vmera' ⊗ Vmera' ⊗ Vmera')
        h = Tensor(f, T, Vmera ⊗ Vmera ⊗ Vmera ⊗ Vmera' ⊗ Vmera' ⊗ Vmera')
        return Timers.Timer(u, w, ρ, h) do u, w, ρ, h
            @tensor C = (
                (
                    (
                        (
                            (((h[9, 3, 4, 5, 1, 2] * u[1, 2, 7, 12]) * conj(u[3, 4, 11, 13])) * (u[8, 5, 15, 6] * w[6, 7, 19])) *
                                (conj(u[8, 9, 17, 10]) * conj(w[10, 11, 22]))
                        ) * ((w[12, 14, 20] * conj(w[13, 14, 23])) * ρ[18, 19, 20, 21, 22, 23])
                    ) * w[16, 15, 18]
                ) * conj(w[16, 17, 21])
            )
            return C
        end
    end
end
