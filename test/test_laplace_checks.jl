# Laplace transforms: GIAC's answers checked against the defining integral, and the transforms it
# misses found term by term

@testset "Laplace transforms" begin
    @giac_var t s a
    C = Giac.Commands
    L(f) = Giac.invoke_cmd(:laplace, f, t, s)
    IL(F) = Giac.invoke_cmd(:ilaplace, F, s, t)
    at(e, x, v) = substitute(e, x => v)
    Γ(p) = Giac.invoke_cmd(:Gamma, p)
    π_ = convert(GiacExpr, π)

    @testset "transforms GIAC misses or gets wrong" begin
        @test at(L(sqrt(t)), s, 1.7) ≈ at(sqrt(π_) / (2s^(3 // 2)), s, 1.7)            # GIAC: 0
        @test at(L(1 / sqrt(t)), s, 1.7) ≈ at(sqrt(π_ / s), s, 1.7)                     # GIAC: 0
        erfc_half = exp(s^2 / 4) * C.erfc(s / 2)
        @test at(L(C.erf(t)), s, 1.7) ≈ at(erfc_half / s, s, 1.7)                       # GIAC: erf for erfc
        @test at(L(exp(-t^2)), s, 1.7) ≈ at(sqrt(π_) / 2 * erfc_half, s, 1.7)
        @test at(L(sqrt(t) * exp(2t)), s, 3.5) ≈ at(sqrt(π_) / (2(s - 2)^(3 // 2)), s, 3.5)
        @test occursin("laplace", string(L(Giac.invoke_cmd(:Si, t))))                   # held, not 0
        @test Giac._laplace_verdict(C.erf(t), t, -C.erf(s / 2) * exp(s^2 / 4) / s, s) === :disagree
    end

    @testset "GIAC's own transforms" begin
        @test string(L(sin(t))) == "1/(s^2+1)"
        @test L(exp(a * t)) == 1 / (s - a)
        @test string(IL(1 / (s^2 + 1))) == "sin(t)"
    end

    @testset "inverses of powers and delays" begin
        @test at(IL(1 / sqrt(s)), t, 0.8) ≈ at(1 / sqrt(π_ * t), t, 0.8)                 # GIAC: times Dirac(t)
        @test at(IL((s + 1)^(-5 // 2)), t, 1.3) ≈ at(t^(3 // 2) * exp(-t) / Γ(5 // 2), t, 1.3)
        delayed = IL(exp(-2s) / s^(3 // 2))
        @test at(delayed, t, 3) ≈ 2 / sqrt(π_) && iszero(at(delayed, t, 1))
        @test occursin("ilaplace", string(IL(1 / (s * sqrt(s + 1)))))                     # held
    end
end
