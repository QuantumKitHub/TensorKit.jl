# TODO: Make into proper tests and integrate in testset
# note: this is not part of the testsuite!

import TensorKit: BraidingTensor

V1 = GradedSpace{FermionSpin}(0 => 2, 1 / 2 => 2, 1 => 1, 3 / 2 => 1)

V2 = GradedSpace{FibonacciAnyon}(:I => 2, :τ => 2)

V3 = GradedSpace{IsingAnyon}(:I => 2, :psi => 1, :sigma => 1)

for V in (V1, V2, V3)
    @show V

    t = randn(V * V' * V' * V, V * V')

    ττ = TensorMap(BraidingTensor(V, V'))
    @planar2 t1[-1 -2 -3 -4; -5 -6] := τ[-1 -2; 1 2] * t[1 2 -3 -4; -5 -6]
    @planar2 t2[-1 -2 -3 -4; -5 -6] := ττ[-1 -2; 1 2] * t[1 2 -3 -4; -5 -6]
    @planar2 t3[-1 -2 -3 -4; -5 -6] := τ[2 1; -2 -1] * t[1 2 -3 -4; -5 -6]
    @planar2 t4[-1 -2 -3 -4; -5 -6] := τ'[-2 2; -1 1] * t[1 2 -3 -4; -5 -6]
    @show norm(t1 - t2), norm(t1 - t3), norm(t1 - t4)

    ττ = TensorMap(BraidingTensor(V', V'))
    @planar2 t1[-1 -2 -3 -4; -5 -6] := τ[-2 -3; 1 2] * t[-1 1 2 -4; -5 -6]
    @planar2 t2[-1 -2 -3 -4; -5 -6] := ττ[-2 -3; 1 2] * t[-1 1 2 -4; -5 -6]
    @planar2 t3[-1 -2 -3 -4; -5 -6] := τ[2 1; -3 -2] * t[-1 1 2 -4; -5 -6]
    @planar2 t4[-1 -2 -3 -4; -5 -6] := τ'[-3 2; -2 1] * t[-1 1 2 -4; -5 -6]
    @show norm(t1 - t2), norm(t1 - t3), norm(t1 - t4)

    ττ = TensorMap(BraidingTensor(V', V))
    @planar2 t1[-1 -2 -3 -4; -5 -6] := τ[1 -2; 2 -3] * t[-1 1 2 -4; -5 -6]
    @planar2 t2[-1 -2 -3 -4; -5 -6] := ττ[1 -2; 2 -3] * t[-1 1 2 -4; -5 -6]
    @planar2 t3[-1 -2 -3 -4; -5 -6] := τ[-3 2; -2 1] * t[-1 1 2 -4; -5 -6]
    @planar2 t4[-1 -2 -3 -4; -5 -6] := τ'[-2 -3; 1 2] * t[-1 1 2 -4; -5 -6]
    @show norm(t1 - t2), norm(t1 - t3), norm(t1 - t4)

    ττ = TensorMap(BraidingTensor(V, V'))
    @planar2 t1[-1 -2 -3 -4; -5 -6] := τ[-3 2; -2 1] * t[-1 1 2 -4; -5 -6]
    @planar2 t2[-1 -2 -3 -4; -5 -6] := ττ[-3 2; -2 1] * t[-1 1 2 -4; -5 -6]
    @planar2 t3[-1 -2 -3 -4; -5 -6] := τ[1 -2; 2 -3] * t[-1 1 2 -4; -5 -6]
    @planar2 t4[-1 -2 -3 -4; -5 -6] := τ'[2 1; -3 -2] * t[-1 1 2 -4; -5 -6]
    @show norm(t1 - t2), norm(t1 - t3), norm(t1 - t4)

    ττ = TensorMap(BraidingTensor(V, V))
    @planar2 t1[-1 -2 -3 -4; -5 -6] := τ[2 1; -3 -2] * t[-1 1 2 -4; -5 -6]
    @planar2 t2[-1 -2 -3 -4; -5 -6] := ττ[2 1; -3 -2] * t[-1 1 2 -4; -5 -6]
    @planar2 t3[-1 -2 -3 -4; -5 -6] := τ[-2 -3; 1 2] * t[-1 1 2 -4; -5 -6]
    @planar2 t4[-1 -2 -3 -4; -5 -6] := τ'[1 -2; 2 -3] * t[-1 1 2 -4; -5 -6]
    @show norm(t1 - t2), norm(t1 - t3), norm(t1 - t4)

    ττ = TensorMap(BraidingTensor(V', V))
    @planar2 t1[-1 -2 -3 -4; -5 -6] := τ[-3 -4; 1 2] * t[-1 -2 1 2; -5 -6]
    @planar2 t2[-1 -2 -3 -4; -5 -6] := ττ[-3 -4; 1 2] * t[-1 -2 1 2; -5 -6]
    @planar2 t3[-1 -2 -3 -4; -5 -6] := τ[2 1; -4 -3] * t[-1 -2 1 2; -5 -6]
    @planar2 t4[-1 -2 -3 -4; -5 -6] := τ'[-4 2; -3 1] * t[-1 -2 1 2; -5 -6]
    @show norm(t1 - t2), norm(t1 - t3), norm(t1 - t4)

    ττ = TensorMap(BraidingTensor(V', V))
    @planar2 t1[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ[1 2; -5 -6]
    @planar2 t2[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * ττ[1 2; -5 -6]
    @planar2 t3[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ[-6 -5; 2 1]
    @planar2 t4[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ'[2 -6; 1 -5]
    @show norm(t1 - t2), norm(t1 - t3), norm(t1 - t4)

    ττ = TensorMap(BraidingTensor(V, V'))
    @planar2 t1[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ'[1 2; -5 -6]
    @planar2 t2[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * ττ'[1 2; -5 -6]
    @planar2 t3[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ'[-6 -5; 2 1]
    @planar2 t4[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ[2 -6; 1 -5]
    @show norm(t1 - t2), norm(t1 - t3), norm(t1 - t4)

    ττ = TensorMap(BraidingTensor(V, V))
    @planar2 t1[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 1; -5 2] * τ[-4 -6; 1 2]
    @planar2 t2[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 1; -5 2] * ττ[-4 -6; 1 2]
    @planar2 t3[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 1; -5 2] * τ[2 1; -6 -4]
    @planar2 t4[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 1; -5 2] * τ'[-6 2; -4 1]
    @show norm(t1 - t2), norm(t1 - t3), norm(t1 - t4)

    ττ = TensorMap(BraidingTensor(V', V))
    @planar2 t1[(); (-1, -2)] := τ[2 1; 3 4] * t[1 2 3 4; -1 -2]
    @planar2 t2[(); (-1, -2)] := ττ[2 1; 3 4] * t[1 2 3 4; -1 -2]
    @planar2 t3[(); (-1, -2)] := τ[4 3; 1 2] * t[1 2 3 4; -1 -2]
    @planar2 t4[(); (-1, -2)] := τ'[1 4; 2 3] * t[1 2 3 4; -1 -2]
    @show norm(t1 - t2), norm(t1 - t3), norm(t1 - t4)

    ττ = TensorMap(BraidingTensor(V, V))
    @planar2 t1[-1; -2] := τ[2 1; 3 4] * t[-1 1 2 3; -2 4]
    @planar2 t2[-1; -2] := ττ[2 1; 3 4] * t[-1 1 2 3; -2 4]
    @planar2 t3[-1; -2] := τ[4 3; 1 2] * t[-1 1 2 3; -2 4]
    @planar2 t4[-1; -2] := τ'[1 4; 2 3] * t[-1 1 2 3; -2 4]
    @show norm(t1 - t2), norm(t1 - t3), norm(t1 - t4)

    ττ = TensorMap(BraidingTensor(V, V'))
    @planar2 t1[-1 -2] := τ[2 1; 3 4] * t[-1 -2 1 2; 4 3]
    @planar2 t2[-1 -2] := ττ[2 1; 3 4] * t[-1 -2 1 2; 4 3]
    @planar2 t3[-1 -2] := τ[4 3; 1 2] * t[-1 -2 1 2; 4 3]
    @planar2 t4[-1 -2] := τ'[1 4; 2 3] * t[-1 -2 1 2; 4 3]
    @show norm(t1 - t2), norm(t1 - t3), norm(t1 - t4)

    ττ = TensorMap(BraidingTensor(V, V'))
    @planar2 t1[-1 -2; -3 -4] := τ[-1 3; 1 2] * t[1 2 3 -2; -3 -4]
    @planar2 t2[-1 -2; -3 -4] := ττ[-1 3; 1 2] * t[1 2 3 -2; -3 -4]
    @planar2 t3[-1 -2; -3 -4] := τ[2 1; 3 -1] * t[1 2 3 -2; -3 -4]
    @planar2 t4[-1 -2; -3 -4] := τ'[3 2; -1 1] * t[1 2 3 -2; -3 -4]
    @show norm(t1 - t2), norm(t1 - t3), norm(t1 - t4)

    ττ = TensorMap(BraidingTensor(V', V'))
    @planar2 t1[-1 -2; -3 -4] := τ'[-2 3; 1 2] * t[-1 1 2 3; -3 -4]
    @planar2 t2[-1 -2; -3 -4] := ττ'[-2 3; 1 2] * t[-1 1 2 3; -3 -4]
    @planar2 t3[-1 -2; -3 -4] := τ'[2 1; 3 -2] * t[-1 1 2 3; -3 -4]
    @planar2 t4[-1 -2; -3 -4] := τ[3 2; -2 1] * t[-1 1 2 3; -3 -4]
    @show norm(t1 - t2), norm(t1 - t3), norm(t1 - t4)

    ττ = TensorMap(BraidingTensor(V', V))
    @planar2 t1[-1 -2 -3; -4] := τ[-3 3; 1 2] * t[-1 -2 1 2; -4 3]
    @planar2 t2[-1 -2 -3; -4] := ττ[-3 3; 1 2] * t[-1 -2 1 2; -4 3]
    @planar2 t3[-1 -2 -3; -4] := τ[2 1; 3 -3] * t[-1 -2 1 2; -4 3]
    @planar2 t4[-1 -2 -3; -4] := τ'[3 2; -3 1] * t[-1 -2 1 2; -4 3]
    @show norm(t1 - t2), norm(t1 - t3), norm(t1 - t4)

    ττ = TensorMap(BraidingTensor(V', V))
    @planar2 t1[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * τ[1 2; -4 3]
    @planar2 t2[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * ττ[1 2; -4 3]
    @planar2 t3[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * τ[3 -4; 2 1]
    @planar2 t4[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * τ'[2 3; 1 -4]
    @show norm(t1 - t2), norm(t1 - t3), norm(t1 - t4)

    ττ = TensorMap(BraidingTensor(V, V'))
    @planar2 t1[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * τ'[1 2; -4 3]
    @planar2 t2[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * ττ'[1 2; -4 3]
    @planar2 t3[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * τ'[3 -4; 2 1]
    @planar2 t4[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * τ[2 3; 1 -4]
    @show norm(t1 - t2), norm(t1 - t3), norm(t1 - t4)
end
