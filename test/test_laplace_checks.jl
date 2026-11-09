# Laplace transforms: GIAC's answers checked against the defining integral, and the transforms it
# misses found term by term

import SpecialFunctions

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
        @test occursin("laplace", string(L(Giac.invoke_cmd(:Ci, t) * sqrt(t))))          # held
        for f in (exp(t^2), t^t, sin(t) * exp(t^2), exp(t^(3 // 2)))                    # no transform
            @test occursin("laplace", string(L(f)))
        end
        @test at(L(exp(5t - 100) * sqrt(t)), s, 7.0) ≈ at(exp(convert(GiacExpr, -100)) * sqrt(π_) / (2(s - 5)^(3 // 2)), s, 7.0)
        @test Giac.Rules.expression_size(L(exp(-t^2))) < 40
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

    @testset "Bessel functions" begin
        J, I, K, Y = SpecialFunctions.besselj, SpecialFunctions.besseli, SpecialFunctions.besselk, SpecialFunctions.bessely
        @giac_var ν
        @test L(J(0, t)) == 1 / sqrt(s^2 + 1)
        @test L(J(ν, a * t)) == (sqrt(s^2 + a^2) - s)^ν / (a^ν * sqrt(s^2 + a^2))
        @test L(I(0, 2t)) == 1 / sqrt(s^2 - 4)
        @test L(J(1, t) / t) == sqrt(s^2 + 1) - s
        @test L(t * J(0, t)) == s / (s^2 + 1)^(3 // 2)
        @test L(J(0, 2sqrt(t))) == exp(-1 / s) / s
        @test at(L(K(0, t)), s, 1.7) ≈ at(log(s + sqrt(s^2 - 1)) / sqrt(s^2 - 1), s, 1.7)
        @test at(L(Y(0, t)), s, 1.7) ≈ at(-2asinh(s) / (π_ * sqrt(s^2 + 1)), s, 1.7)
        @test L(exp(-t) * J(0, t)) == 1 / sqrt((s + 1)^2 + 1)
        @test IL(1 / sqrt(s^2 + 1)) == J(0, t)
        @test IL(1 / sqrt(s^2 - 4)) == I(0, 2t)
        @test IL((s^2 + 1)^(-3 // 2)) == t * J(1, t)
        @test IL(exp(-1 / s) / s) == J(0, 2sqrt(t))
        @test IL(1 / sqrt(s^2 + 2s + 2)) == exp(-t) * J(0, t)
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
