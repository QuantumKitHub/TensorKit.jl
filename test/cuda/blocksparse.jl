using Test, TestExtras
using CUDA, cuTENSOR
using TensorKit
using TensorOperations: TensorOperations as TO
using TensorKit: BlockSparseUnsupported, blocksparsestructure

@isdefined(TestSetup) || include("../setup.jl")
using .TestSetup

const CUDAExt = Base.get_extension(TensorKit, :TensorKitCUDAExt)
const cuTENSORExt = Base.get_extension(TensorKit, :TensorKitcuTENSORExt)
@assert !isnothing(CUDAExt) && !isnothing(cuTENSORExt)
const CuTensorMap = getglobal(CUDAExt, :CuTensorMap)
const blocksparse_supported = getglobal(cuTENSORExt, :blocksparse_supported)
const blocksparse_reason = getglobal(cuTENSORExt, :blocksparse_reason)
const check_plan = getglobal(cuTENSORExt, :check_plan)
const _signs_index = getglobal(cuTENSORExt, :_signs_index)
const bs_signs = getglobal(TensorKit, :blocksparse_contract_signs)
const PLAN_CACHE = getglobal(cuTENSORExt, :BLOCKSPARSE_PLAN_CACHE)

const bs = CuTENSORBlockSparse()
const bs_strict = CuTENSORBlockSparse(; strict = true)

# supported abelian spaces, all of which contain dual legs
const abelian_spaces = (
    "ℤ₂" => TestSetup.VRepℤ₂,
    "ℤ₃" => TestSetup.VRepℤ₃,
    "U₁" => TestSetup.VRepU₁,
)
# supported fermionic spaces; `TestSetup.VfHubbard` has an `SU2Irrep` factor, so it cannot serve
const VfRepU₁ = (
    Vect[FermionNumber](0 => 2, 1 => 2, -1 => 1),
    Vect[FermionNumber](0 => 1, 1 => 1, -1 => 1)',
    Vect[FermionNumber](0 => 2, 1 => 1, -1 => 2),
    Vect[FermionNumber](0 => 1, 1 => 2, -1 => 1)',
    Vect[FermionNumber](0 => 1, 1 => 1, -1 => 2),
)
const fermionic_spaces = ("fℤ₂" => TestSetup.VfRepℤ₂, "fU₁" => VfRepU₁)
const supported_spaces = (abelian_spaces..., fermionic_spaces...)
const eltypes = (Float32, Float64, ComplexF32, ComplexF64)

"contract twice -- once through the default path, once block-sparse -- and compare raw data"
function compare_paths(V, T, pA, pB, pAB, α, β; conjA = false, conjB = false)
    V1, V2, V3, V4, V5 = V
    WA = V1 ⊗ V2 ← V3
    # conjugating exactly one operand dualizes the contracted leg's space requirement
    WB = (conjA ⊻ conjB) ? (dual(V3) ← V4 ⊗ V5) : (V3 ← V4 ⊗ V5)
    A = CuTensorMap(randn(T, WA))
    B = CuTensorMap(randn(T, WB))
    WC = TO.tensorcontract(space(A), pA, conjA, space(B), pB, conjB, pAB)
    C0 = CuTensorMap(randn(T, WC))

    ref = copy(C0)
    TO.tensorcontract!(ref, A, pA, conjA, B, pB, conjB, pAB, α, β, TO.DefaultBackend(), TO.DefaultAllocator())
    got = copy(C0)
    TO.tensorcontract!(got, A, pA, conjA, B, pB, conjB, pAB, α, β, bs_strict, TO.DefaultAllocator())
    return Array(got.data), Array(ref.data)
end

@testset "blocksparse vs default path: $name, $T" for (name, V) in supported_spaces,
        T in eltypes

    V1, V2, V3, V4, V5 = V
    # A: V1 ⊗ V2 ← V3, B: V3 ← V4 ⊗ V5, contract index 3 of A with index 1 of B
    pA = ((1, 2), (3,))
    pB = ((1,), (2, 3))
    for (pAB, label) in (
            (((1, 2), (3, 4)), "natural"),
            (((1, 2, 3, 4), ()), "all in codomain"),
            (((), (1, 2, 3, 4)), "all in domain"),
            (((2, 1), (4, 3)), "permuted output"),
            (((3, 1), (2, 4)), "interleaved output"),
        )
        for (α, β) in ((one(T), zero(T)), (T(2), zero(T)), (one(T), one(T)), (T(-1.5), T(0.5)))
            got, ref = compare_paths(V, T, pA, pB, pAB, α, β)
            @test got ≈ ref
        end
    end
    # α, β genuinely complex
    if T <: Complex
        got, ref = compare_paths(V, T, pA, pB, ((1, 2), (3, 4)), T(1 + 2im), T(0.5 - 1im))
        @test got ≈ ref
    end
end

@testset "contracted legs in other positions: $name" for (name, V) in supported_spaces
    V1, V2, V3, V4, V5 = V
    T = ComplexF64
    # contract A's *codomain* index 1 with an index sitting in B's *domain*
    A = CuTensorMap(randn(T, V1 ⊗ V2 ← V3))
    B = CuTensorMap(randn(T, V4 ← V1 ⊗ V5))
    pA, pB, pAB = ((2, 3), (1,)), ((2,), (1, 3)), ((1, 2), (3, 4))
    WC = TO.tensorcontract(space(A), pA, false, space(B), pB, false, pAB)
    C0 = CuTensorMap(randn(T, WC))
    ref, got = copy(C0), copy(C0)
    TO.tensorcontract!(ref, A, pA, false, B, pB, false, pAB, one(T), zero(T), TO.DefaultBackend(), TO.DefaultAllocator())
    TO.tensorcontract!(got, A, pA, false, B, pB, false, pAB, one(T), zero(T), bs_strict, TO.DefaultAllocator())
    @test Array(got.data) ≈ Array(ref.data)
end

@testset "conjugation: $name, $T" for (name, V) in supported_spaces, T in (Float64, ComplexF64)
    for (conjA, conjB) in ((true, false), (false, true), (true, true))
        # a no-op elementwise for real scalars, but it still changes the effective permutation
        for pAB in (((1, 2), (3, 4)), ((2, 1), (4, 3)), ((3, 1), (2, 4)))
            got, ref = compare_paths(
                V, T, ((1, 2), (3,)), ((1,), (2, 3)), pAB, one(T), zero(T); conjA, conjB
            )
            @test got ≈ ref
        end
    end
end

# contract over the dual `V4`, so that `blas_contract!` does insert a twist
@testset "dual contracted leg: $name, $T" for (name, V) in supported_spaces,
        T in (Float64, ComplexF64)

    V1, V2, V3, V4, V5 = V
    A = CuTensorMap(randn(T, V1 ⊗ V2 ← V4))
    B = CuTensorMap(randn(T, V4 ← V3 ⊗ V5))
    pA, pB = ((1, 2), (3,)), ((1,), (2, 3))
    for pAB in (((1, 2), (3, 4)), ((2, 1), (4, 3)), ((3, 1), (2, 4)), ((1, 2, 3, 4), ()))
        WC = TO.tensorcontract(space(A), pA, false, space(B), pB, false, pAB)
        C0 = CuTensorMap(randn(T, WC))
        for (α, β) in ((one(T), zero(T)), (T(-1.5), T(0.5)))
            ref, got = copy(C0), copy(C0)
            TO.tensorcontract!(ref, A, pA, false, B, pB, false, pAB, α, β, TO.DefaultBackend(), TO.DefaultAllocator())
            TO.tensorcontract!(got, A, pA, false, B, pB, false, pAB, α, β, bs_strict, TO.DefaultAllocator())
            @test Array(got.data) ≈ Array(ref.data)
        end
    end
end

@testset "multiple contracted indices: $name" for (name, V) in supported_spaces
    V1, V2, V3, V4, V5 = V
    T = Float64
    A = CuTensorMap(randn(T, V1 ← V2 ⊗ V3))
    B = CuTensorMap(randn(T, V2 ⊗ V3 ← V4))
    pA, pB, pAB = ((1,), (2, 3)), ((1, 2), (3,)), ((1,), (2,))
    WC = TO.tensorcontract(space(A), pA, false, space(B), pB, false, pAB)
    C0 = CuTensorMap(randn(T, WC))
    ref, got = copy(C0), copy(C0)
    TO.tensorcontract!(ref, A, pA, false, B, pB, false, pAB, one(T), zero(T), TO.DefaultBackend(), TO.DefaultAllocator())
    TO.tensorcontract!(got, A, pA, false, B, pB, false, pAB, one(T), zero(T), bs_strict, TO.DefaultAllocator())
    @test Array(got.data) ≈ Array(ref.data)
    # and with the contracted indices in the other order, which forces different mode labels
    pA2, pB2 = ((1,), (3, 2)), ((2, 1), (3,))
    ref2, got2 = copy(C0), copy(C0)
    TO.tensorcontract!(ref2, A, pA2, false, B, pB2, false, pAB, one(T), zero(T), TO.DefaultBackend(), TO.DefaultAllocator())
    TO.tensorcontract!(got2, A, pA2, false, B, pB2, false, pAB, one(T), zero(T), bs_strict, TO.DefaultAllocator())
    @test Array(got2.data) ≈ Array(ref2.data)
end

@testset "@tensor integration: $name" for (name, V) in (
        "U₁" => TestSetup.VRepU₁, "fℤ₂" => TestSetup.VfRepℤ₂,
    )
    V1, V2, V3, V4, V5 = V
    A = CuTensorMap(randn(ComplexF64, V1 ⊗ V2 ← V3))
    B = CuTensorMap(randn(ComplexF64, V3 ← V4 ⊗ V5))
    # `@tensor` is the only route supplying `α = One()`, `β = Zero()`/`One()`
    @tensor ref[a, b, d, e] := A[a, b, c] * B[c, d, e]
    @tensor backend = bs got[a, b, d, e] := A[a, b, c] * B[c, d, e]
    @test Array(got.data) ≈ Array(ref.data)
    # `+=` gives `β = One()`, which takes the pre-scaling path
    @tensor ref[a, b, d, e] += A[a, b, c] * B[c, d, e]
    @tensor backend = bs got[a, b, d, e] += A[a, b, c] * B[c, d, e]
    @test Array(got.data) ≈ Array(ref.data)
    # an output permutation, so that the result correction is non-trivial for fermions
    @tensor ref2[b, a, e, d] := A[a, b, c] * B[c, d, e]
    @tensor backend = bs got2[b, a, e, d] := A[a, b, c] * B[c, d, e]
    @test Array(got2.data) ≈ Array(ref2.data)
end

@testset "shared operand: $name" for (name, V) in supported_spaces
    # `A === B` is legal, so the sign correction must not scale a shared operand in place
    V1, V2, V3, V4, V5 = V
    T = ComplexF64
    A = CuTensorMap(randn(T, V3 ← V3 ⊗ V3))
    pA, pB = ((1, 2), (3,)), ((1,), (2, 3))
    for pAB in (((1, 2), (3, 4)), ((2, 1), (4, 3)))
        WC = TO.tensorcontract(space(A), pA, false, space(A), pB, false, pAB)
        C0 = CuTensorMap(randn(T, WC))
        ref, got = copy(C0), copy(C0)
        Adata = Array(A.data)
        TO.tensorcontract!(ref, A, pA, false, A, pB, false, pAB, one(T), zero(T), TO.DefaultBackend(), TO.DefaultAllocator())
        TO.tensorcontract!(got, A, pA, false, A, pB, false, pAB, one(T), zero(T), bs_strict, TO.DefaultAllocator())
        @test Array(got.data) ≈ Array(ref.data)
        @test Array(A.data) == Adata     # ... and the operand is left untouched
    end
end

@testset "fallback is silent and correct" begin
    # non-abelian and `Trivial` sectors are unsupported, but must still give the right answer
    for V in (TestSetup.VRepSU₂, TestSetup.VfHubbard, TestSetup.Vtr)
        V1, V2, V3, V4, V5 = V
        T = ComplexF64
        A = CuTensorMap(randn(T, V1 ⊗ V2 ← V3))
        B = CuTensorMap(randn(T, V3 ← V4 ⊗ V5))
        pA, pB, pAB = ((1, 2), (3,)), ((1,), (2, 3)), ((1, 2), (3, 4))
        WC = TO.tensorcontract(space(A), pA, false, space(B), pB, false, pAB)
        C0 = CuTensorMap(randn(T, WC))
        ref, got = copy(C0), copy(C0)
        TO.tensorcontract!(ref, A, pA, false, B, pB, false, pAB, one(T), zero(T), TO.DefaultBackend(), TO.DefaultAllocator())
        TO.tensorcontract!(got, A, pA, false, B, pB, false, pAB, one(T), zero(T), bs, TO.DefaultAllocator())
        @test Array(got.data) ≈ Array(ref.data)
        @test !blocksparse_supported(C0, A, pA, false, B, pB, false, pAB)
        # ... and `strict` complains instead
        @test_throws BlockSparseUnsupported TO.tensorcontract!(
            copy(C0), A, pA, false, B, pB, false, pAB,
            one(T), zero(T), bs_strict, TO.DefaultAllocator()
        )
    end
end

@testset "unsupported eltype falls back" begin
    V = TestSetup.VRepℤ₂
    V1, V2, V3, V4, V5 = V
    T = ComplexF16
    A = CuTensorMap(randn(T, V1 ⊗ V2 ← V3))
    B = CuTensorMap(randn(T, V3 ← V4 ⊗ V5))
    pA, pB, pAB = ((1, 2), (3,)), ((1,), (2, 3)), ((1, 2), (3, 4))
    WC = TO.tensorcontract(space(A), pA, false, space(B), pB, false, pAB)
    C0 = CuTensorMap(randn(T, WC))
    @test !blocksparse_supported(C0, A, pA, false, B, pB, false, pAB)
    @test occursin("scalar type", blocksparse_reason(C0, A, pA, false, B, pB, false, pAB))
end

@testset "a failed contraction does not corrupt the output" begin
    # a complex `α` throws before cuTENSOR touches anything, so the output correction must not
    # be left half-applied -- for `β = 0` that would negate every block whose sign is -1
    V1, V2, V3, V4, V5 = TestSetup.VfRepℤ₂
    T = Float64
    A = CuTensorMap(randn(T, V1 ⊗ V2 ← V3))
    B = CuTensorMap(randn(T, V3 ← V4 ⊗ V5))
    pA, pB, pAB = ((1, 2), (3,)), ((1,), (2, 3)), ((2, 1), (4, 3))   # non-trivial `signs.C`
    WC = TO.tensorcontract(space(A), pA, false, space(B), pB, false, pAB)
    C = CuTensorMap(randn(T, WC))
    for β in (zero(T), one(T))
        before = Array(C.data)
        @test_throws InexactError TO.tensorcontract!(
            C, A, pA, false, B, pB, false, pAB, T(1) + 2im, β, bs_strict,
            TO.DefaultAllocator()
        )
        @test Array(C.data) == before
    end
end

@testset "a plan carries the sign correction for every conjugation" begin
    # `_signs_index` must agree with the order `_create_plan` derives the combinations in
    V1, V2, V3, V4, V5 = TestSetup.VfRepℤ₂
    T = ComplexF64
    pA, pB, pAB = ((1, 2), (3,)), ((1,), (2, 3)), ((2, 1), (4, 3))
    WA, WB = V1 ⊗ V2 ← V3, V3 ← V4 ⊗ V5
    WC = TO.tensorcontract(WA, pA, false, WB, pB, false, pAB)
    plan = plan_contract(T, WC, WA, pA, WB, pB, pAB)
    @test length(plan.signs) == 4
    for conjA in (false, true), conjB in (false, true)
        expected = bs_signs(WC, WA, pA, conjA, WB, pB, conjB, pAB)
        got = plan.signs[_signs_index(conjA, conjB)]
        @test (got.A, got.B, got.C) == (expected.A, expected.B, expected.C)
    end
    # a bosonic plan carries four trivial corrections, i.e. costs nothing to have derived
    Vb = TestSetup.VRepU₁
    WAb, WBb = Vb[1] ⊗ Vb[2] ← Vb[3], Vb[3] ← Vb[4] ⊗ Vb[5]
    WCb = TO.tensorcontract(WAb, pA, false, WBb, pB, false, pAB)
    planb = plan_contract(T, WCb, WAb, pA, WBb, pB, pAB)
    @test all(TensorKit.istrivial, planb.signs)
end

@testset "plan_contract rejects unsupported sectors" begin
    # a plan never consults `blocksparse_reason`, so the sector gate is applied when it is built
    V1, V2, V3, V4, V5 = TestSetup.VfHubbard
    T = ComplexF64
    A = CuTensorMap(randn(T, V1 ⊗ V2 ← V3))
    B = CuTensorMap(randn(T, V3 ← V4 ⊗ V5))
    pA, pB, pAB = ((1, 2), (3,)), ((1,), (2, 3)), ((1, 2), (3, 4))
    WC = TO.tensorcontract(space(A), pA, false, space(B), pB, false, pAB)
    C = CuTensorMap(randn(T, WC))
    @test_throws BlockSparseUnsupported plan_contract(C, A, pA, B, pB, pAB)

    # a plan bypasses `blocksparse_reason`, so it must refuse an aliased output itself
    V = TestSetup.VfRepℤ₂[3]
    A2 = CuTensorMap(randn(T, V ← V ⊗ V))
    WC2 = TO.tensorcontract(space(A2), pA, false, space(A2), pB, false, pAB)
    C2 = CuTensorMap(randn(T, WC2))
    plan = plan_contract(C2, A2, pA, A2, pB, pAB)
    @test isnothing(check_plan(plan, C2, A2, pA, A2, pB, pAB))
    @test_throws ArgumentError check_plan(plan, A2, A2, pA, A2, pB, pAB)
end

@testset "plan reuse" begin
    V = TestSetup.VRepU₁
    V1, V2, V3, V4, V5 = V
    T = Float64
    pA, pB, pAB = ((1, 2), (3,)), ((1,), (2, 3)), ((1, 2), (3, 4))
    A = CuTensorMap(randn(T, V1 ⊗ V2 ← V3))
    B = CuTensorMap(randn(T, V3 ← V4 ⊗ V5))
    WC = TO.tensorcontract(space(A), pA, false, space(B), pB, false, pAB)
    C = CuTensorMap(randn(T, WC))

    empty!(PLAN_CACHE)
    TO.tensorcontract!(C, A, pA, false, B, pB, false, pAB, one(T), zero(T), bs_strict, TO.DefaultAllocator())
    @test length(PLAN_CACHE) == 1
    # same spaces, different data: still one plan
    A2 = CuTensorMap(randn(T, V1 ⊗ V2 ← V3))
    TO.tensorcontract!(C, A2, pA, false, B, pB, false, pAB, one(T), zero(T), bs_strict, TO.DefaultAllocator())
    @test length(PLAN_CACHE) == 1

    # an explicitly supplied plan gives identical results
    plan = plan_contract(C, A, pA, B, pB, pAB)
    ref, got = copy(C), copy(C)
    TO.tensorcontract!(ref, A, pA, false, B, pB, false, pAB, T(1.5), T(-0.25), bs_strict, TO.DefaultAllocator())
    TO.tensorcontract!(got, A, pA, false, B, pB, false, pAB, T(1.5), T(-0.25), plan, TO.DefaultAllocator())
    @test Array(got.data) ≈ Array(ref.data)
    # the data-free form agrees
    plan2 = plan_contract(T, space(C), space(A), pA, space(B), pB, pAB)
    @test plan2 === plan

    # reusing a plan for a different (but individually valid) contraction is caught
    Vs = Vect[U1Irrep](0 => 1, 1 => 1, -1 => 1)
    A3 = CuTensorMap(randn(T, Vs ⊗ Vs ← Vs))
    B3 = CuTensorMap(randn(T, Vs ← Vs ⊗ Vs))
    WC3 = TO.tensorcontract(space(A3), pA, false, space(B3), pB, false, pAB)
    C3 = CuTensorMap(randn(T, WC3))
    @test_throws ArgumentError TO.tensorcontract!(
        C3, A3, pA, false, B3, pB, false, pAB,
        one(T), zero(T), plan, TO.DefaultAllocator()
    )

    # the cache is registered with TensorKit's cache registry
    empty_globalcaches!()
    @test isempty(PLAN_CACHE)
    TO.tensorcontract!(C, A, pA, false, B, pB, false, pAB, one(T), zero(T), bs_strict, TO.DefaultAllocator())
    @test length(PLAN_CACHE) == 1
end

@testset "non-trivial workspace" begin
    # large enough that cuTENSOR asks for a real workspace, which the wrapper's argtype bug broke
    V = Vect[U1Irrep](-1 => 24, 0 => 32, 1 => 24)
    T = Float64
    pA, pB, pAB = ((1, 2), (3,)), ((1,), (2, 3)), ((1, 2), (3, 4))
    A = CuTensorMap(randn(T, V ⊗ V ← V))
    B = CuTensorMap(randn(T, V ← V ⊗ V))
    WC = TO.tensorcontract(space(A), pA, false, space(B), pB, false, pAB)
    C = CuTensorMap(randn(T, WC))
    plan = plan_contract(C, A, pA, B, pB, pAB)
    @test plan.workspacesize > 0
    ref, got = copy(C), copy(C)
    TO.tensorcontract!(ref, A, pA, false, B, pB, false, pAB, one(T), zero(T), TO.DefaultBackend(), TO.DefaultAllocator())
    TO.tensorcontract!(got, A, pA, false, B, pB, false, pAB, one(T), zero(T), bs_strict, TO.DefaultAllocator())
    @test Array(got.data) ≈ Array(ref.data)
end
