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
        @test occursin("laplace", string(L(Giac.invoke_cmd(:BesselJ, 0, t))))           # held, not infinity
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
        @test string(IL(exp(-3s) / (s * sqrt(s + 1)))) == "erf(sqrt(t-3))*Heaviside(t-3)"
        @test occursin("ilaplace", string(IL(exp(-s^2))))                                 # held
    end

    @testset "pairs in √t and 1/t" begin
        k = 3
        @test at(L(sin(k * sqrt(t))), s, 1.7) ≈ at(k * sqrt(π_) / (2s^(3 // 2)) * exp(-k^2 / (4s)), s, 1.7)
        @test at(L(cosh(sqrt(t)) / sqrt(t)), s, 1.7) ≈ at(sqrt(π_ / s) * exp(1 / (4s)), s, 1.7)
        @test at(L(exp(-a / t) / t^(3 // 2)), s, 1.7) ≈ at(sqrt(π_ / a) * exp(-2sqrt(a * s)), s, 1.7)
        @test L(C.erfc(1 / sqrt(t))) == exp(-2sqrt(s)) / s
        @test at(IL(exp(-2sqrt(s)) / sqrt(s)), t, 0.8) ≈ at(exp(-1 / t) / sqrt(π_ * t), t, 0.8)
        @test at(IL(exp(-sqrt(s))), t, 0.8) ≈ at(exp(-1 / (4t)) / (2sqrt(π_) * t^(3 // 2)), t, 0.8)
        @test IL(exp(-1 / s) / sqrt(s)) == cos(2sqrt(t)) / sqrt(π_ * t)
        @test IL(exp(1 / s) / s^(3 // 2)) == sinh(2sqrt(t)) / sqrt(π_)
    end

    @testset "transforms from the theorems" begin
        Si = Giac.invoke_cmd(:Si, t)
        @test L(Si) == (π_ / 2 - atan(s)) / s                                           # from Si' = sin(t)/t
        @test at(L(t * Si), s, 1.7) ≈ at(1 / (s * (1 + s^2)) + (π_ / 2 - atan(s)) / s^2, s, 1.7)
        @test L(C.erf(2sqrt(t))) == 2 / (s * sqrt(s + 4))
        @test at(L((1 - cos(t)) / t), s, 1.7) ≈ at(log(1 + 1 / s^2) / 2, s, 1.7)
        @test string(IL(1 / (s * sqrt(s + 1)))) == "erf(sqrt(t))"                       # ∫₀ᵗ of the inverse
        @test IL(atan(1 / s)) == sin(t) / t                                              # from the derivative
        @test at(IL(log((s + 1) / (s + 2))), t, 0.8) ≈ at((exp(-2t) - exp(-t)) / t, t, 0.8)
        @test at(IL(1 / (sqrt(s) * (s - 1))), t, 0.8) ≈ at(exp(t) * C.erf(sqrt(t)), t, 0.8)  # a convolution
    end
end
