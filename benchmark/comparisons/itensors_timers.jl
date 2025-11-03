include("benchtools.jl")
module ITensorsTimers
    using ITensors
    using ..Timers: Timers

    function add_indexlabels_as_tags(tensor, labels)
        new_inds = map(x -> settags(x[2], "l$(x[1])"), zip(labels, tensor.inds))
        return ITensor(tensor.store, new_inds)
    end

    function mpo_timer(T = Float64; Vmpo, Vmps, Vphys)
        A = randomITensor(T, Vmps, Vphys, dag(Vmps))
        M = randomITensor(T, Vmpo, Vphys, dag(Vphys), dag(Vmpo))
        FL = randomITensor(T, Vmps, dag(Vmpo), dag(Vmps))
        FR = randomITensor(T, Vmps, Vmpo, dag(Vmps))

        FL1 = add_indexlabels_as_tags(FL, (4, 2, 1))
        A1 = add_indexlabels_as_tags(A, (1, 3, 6))
        M1 = add_indexlabels_as_tags(M, (2, 5, 3, 7))
        A2 = add_indexlabels_as_tags(A, (4, 5, 8))
        FR1 = add_indexlabels_as_tags(FR, (6, 7, 8))

        return Timers.Timer(A1, A2, M1, FL1, FR1) do A1, A2, M1, FL1, FR1
            C = ((((FL1 * A1) * M1) * dag(A2)) * FR1)
            return C[]
        end
    end

    function pepo_timer(T = Float64; Vpepo, Vpeps, Venv, Vphys)
        Vpeps′ = dag(Vpeps)
        Vpepo′ = dag(Vpepo)
        Venv′ = dag(Venv)
        Vphys′ = dag(Vphys)

        A = randomITensor(T, Vpeps, Vpeps, Vphys, Vpeps′, Vpeps′)
        P = randomITensor(T, Vpepo, Vpepo, Vphys, Vphys′, Vpepo′, Vpepo′)
        FL = randomITensor(T, Venv, Vpeps, Vpepo′, Vpeps′, Venv′)
        FD = randomITensor(T, Venv, Vpeps, Vpepo′, Vpeps′, Venv′)
        FR = randomITensor(T, Venv, Vpeps, Vpepo, Vpeps′, Venv′)
        FU = randomITensor(T, Venv, Vpeps, Vpepo, Vpeps′, Venv′)

        FL1 = add_indexlabels_as_tags(FL, (18, 7, 4, 2, 1))
        FU1 = add_indexlabels_as_tags(FU, (1, 3, 6, 9, 10))
        A1 = add_indexlabels_as_tags(A, (2, 17, 5, 3, 11))
        P1 = add_indexlabels_as_tags(P, (4, 16, 8, 5, 6, 12))
        A2 = add_indexlabels_as_tags(A, (7, 15, 8, 9, 13))
        FR1 = add_indexlabels_as_tags(FR, (10, 11, 12, 13, 14))
        FD1 = add_indexlabels_as_tags(FD, (14, 15, 16, 17, 18))

        return Timers.Timer(A1, A2, P1, FL1, FD1, FR1, FU1) do A1, A2, P1, FL1, FD1, FR1, FU1
            C = (((((FL1 * FU1) * A1) * P1) * dag(A2)) * FR1) * FD1
            return C[]
        end
    end

    function mera_timer(T = Float64; Vmera)
        Vmera′ = dag(Vmera)

        u = randomITensor(T, Vmera, Vmera, Vmera′, Vmera′)
        w = randomITensor(T, Vmera, Vmera, Vmera′)
        ρ = randomITensor(T, Vmera, Vmera, Vmera, Vmera′, Vmera′, Vmera′)
        h = randomITensor(T, Vmera, Vmera, Vmera, Vmera′, Vmera′, Vmera′)

        h1 = add_indexlabels_as_tags(h, (9, 3, 4, 5, 1, 2))
        u1 = add_indexlabels_as_tags(u, (1, 2, 7, 12))
        u2 = add_indexlabels_as_tags(u, (3, 4, 11, 13))
        u3 = add_indexlabels_as_tags(u, (8, 5, 15, 6))
        w1 = add_indexlabels_as_tags(w, (6, 7, 19))
        u4 = add_indexlabels_as_tags(u, (8, 9, 17, 10))
        w2 = add_indexlabels_as_tags(w, (10, 11, 22))
        w3 = add_indexlabels_as_tags(w, (12, 14, 20))
        w4 = add_indexlabels_as_tags(w, (13, 14, 23))
        ρ1 = add_indexlabels_as_tags(ρ, (18, 19, 20, 21, 22, 23))
        w5 = add_indexlabels_as_tags(w, (16, 15, 18))
        w6 = add_indexlabels_as_tags(w, (16, 17, 21))

        return Timers.Timer(
            u1, u2, u3, u4, w1, w2, w3, w4, w5, w6, ρ1, h1
        ) do u1, u2, u3, u4, w1, w2, w3, w4, w5, w6, ρ1, h1
            C = (((((((h1 * u1) * dag(u2)) * (u3 * w1)) * (dag(u4) * dag(w2))) * ((w3 * dag(w4)) * ρ1)) * w5) * dag(w6))
            return C
        end
    end
end
