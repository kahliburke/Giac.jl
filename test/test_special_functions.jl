# Bessel, Hankel and Airy functions: calls GIAC does not interpret, with derivatives on the tree,
# values from SpecialFunctions, and half-integer orders written with elementary functions

import SpecialFunctions as SF

@testset "Bessel and Airy functions" begin
    @giac_var x n
    D(f, k = 1) = Giac.invoke_cmd(:diff, f, x, k)
    two = convert(GiacExpr, 2)
    π_ = convert(GiacExpr, π)

    @testset "orders" begin
        @test SF.besselj(1 // 2, x) == sqrt(2 / (π_ * x)) * sin(x)
        @test SF.besselj(-1 // 2, x) == sqrt(2 / (π_ * x)) * cos(x)
        @test SF.besselj(3 // 2, x) == sqrt(2 / (π_ * x)) * (sin(x) / x - cos(x))
        @test SF.besseli(1 // 2, x) == sqrt(2 / (π_ * x)) * sinh(x)
        @test SF.besselk(5 // 2, x) == sqrt(π_ / (2x)) * exp(-x) * (1 + 3 / x + 3 / x^2)
        @test SF.besselj(-3, x) == -SF.besselj(3, x)
        @test SF.besseli(-2, x) == SF.besseli(2, x)
        @test SF.besselj(0, convert(GiacExpr, 0)) == 1 && SF.besselj(2, convert(GiacExpr, 0)) == 0
        @test string(SF.besselj(n, x)) == "besselj(n,x)"
    end

    @testset "derivatives" begin
        @test D(SF.besselj(n, x)) == (SF.besselj(n - 1, x) - SF.besselj(n + 1, x)) / 2
        @test D(SF.besselj(0, x^2)) == -2x * SF.besselj(1, x^2)
        @test D(SF.besseli(0, x)) == SF.besseli(1, x)
        @test D(SF.besselk(0, x)) == -SF.besselk(1, x)
        @test D(SF.besselk(n, x)) == -(SF.besselk(n - 1, x) + SF.besselk(n + 1, x)) / 2
        @test D(SF.airyai(x)) == SF.airyaiprime(x)
        @test D(SF.airybi(x), 2) == x * SF.airybi(x)
        # Bessel's equation x²y'' + xy' + (x² - n²)y = 0
        y = SF.besselj(n, x)
        @test iszero(Giac.invoke_cmd(:simplify, x^2 * D(y, 2) + x * D(y) + (x^2 - n^2) * y))
    end

    @testset "values" begin
        @test to_julia(SF.besselj(0, two)) == SF.besselj(0, 2.0)
        @test to_julia(SF.hankelh1(1, two)) ≈ SF.hankelh1(1, 2.0)
        @test float(SF.besselk(1 // 3, two) + 1) ≈ SF.besselk(1 / 3, 2.0) + 1
        @test SF.besselj(0, two)^2 + SF.bessely(0, two)^2 ≈ abs2(SF.hankelh1(0, 2.0))
        @test to_julia(Giac.invoke_cmd(:evalf, SF.airyai(two) + 1)) ≈ SF.airyai(2.0) + 1
        @test string(SF.besselj(0, convert(GiacExpr, 2.0))) == string(SF.besselj(0, 2.0))
        f = build_function(SF.besselj(1, x) * exp(-x) + SF.airyai(x), x; backend = :julia)
        @test f(1.3) ≈ SF.besselj(1, 1.3) * exp(-1.3) + SF.airyai(1.3)
        @test isempty(Giac._free_names(SF.besselj(1, two)))
    end

    @testset "antiderivatives" begin
        ∫(f) = Giac.Commands.integrate(f, x)
        @test ∫(x * SF.besselj(0, x)) == x * SF.besselj(1, x)
        @test ∫(SF.besselj(1, x) + x^2) == -SF.besselj(0, x) + x^3 / 3
        @test ∫(x^3 * SF.besselj(0, x)) == x^3 * SF.besselj(1, x) - 2x^2 * SF.besselj(2, x)
        @test ∫(x * SF.besselk(0, 3x)) == -x * SF.besselk(1, 3x) / 3
        @test ∫(x * SF.airyai(x)) == SF.airyaiprime(x)
        for f in (x^3 * SF.besseli(0, 2x), x^2 * SF.bessely(1, x), SF.hankelh1(1, x))
            @test iszero(Giac.invoke_cmd(:simplify, D(∫(f)) - f))
        end
    end

    @testset "Struve functions" begin
        H, L = Giac.struveh, Giac.struvel
        val(e, v) = to_julia(Giac.invoke_cmd(:evalf, substitute(e, x => v)))
        series(ν, v; modified = false) = sum((modified ? 1 : (-1)^k) * (v / 2)^(2k + ν + 1) /
                                             (SF.gamma(k + 3 / 2) * SF.gamma(k + ν + 3 / 2)) for k in 0:30)
        @test H(0, 1.0) ≈ series(0, 1.0) rtol = 1e-14
        @test L(0, 2.0) ≈ series(0, 2.0; modified = true) rtol = 1e-14
        @test H(2.3, 5.0) ≈ series(2.3, 5.0) rtol = 1e-13
        @test H(-0.4, 2.0) ≈ series(-0.4, 2.0) rtol = 1e-13
        @test H(1 // 2, 300.0) ≈ sqrt(2 / (300π)) * (1 - cos(300.0)) rtol = 1e-11
        @test H(0, 100.0) - SF.bessely(0, 100.0) ≈ 2 / (100π) * (1 - 1e-4 + 9e-8) rtol = 1e-8
        @test H(1 // 2, x) == sqrt(2 / (π_ * x)) * (1 - cos(x))
        @test H(-1, x) == 2 / π_ - H(1, x)
        @test val(L(3 // 2, x), 2.3) ≈ L(1.5, 2.3)
        @test D(H(0, x)) == 2 / π_ - H(1, x)
        @test to_julia(H(0, convert(GiacExpr, 1))) ≈ H(0, 1.0)
        @test Giac.giac_latex(H(n, x)) == "\\mathbf{H}_{n}\\left(x\\right)"
        # Bessel functions times even powers integrate to Struve functions
        ∫(f) = Giac.Commands.integrate(f, x)
        @test ∫(SF.besselj(0, x)) == x * SF.besselj(0, x) + π_ * x * (SF.besselj(1, x) * H(0, x) - SF.besselj(0, x) * H(1, x)) / 2
        for f in (x^2 * SF.besselj(0, x), SF.bessely(0, 2x), SF.besseli(0, x), x^4 * SF.besseli(0, x), SF.hankelh1(0, x))
            @test abs(val(D(∫(f)) - f, 1.7)) < 1e-11
        end
    end

    @testset "definite integrals" begin
        ∞ = convert(GiacExpr, Inf)
        ∫(f, a = 0, b = ∞) = Giac.Commands.integrate(f, x, a, b)
        @test ∫(SF.besselj(0, x)) == 1
        @test ∫(SF.besselj(1, x) / x) == 1
        @test ∫(SF.besselj(0, 2x)) == 1 // 2
        @test ∫(exp(-x) * SF.besselj(0, x)) == 1 / sqrt(two)
        @test ∫(SF.besselk(0, x)) == π_ / 2
        @test ∫(exp(-2x) * SF.besseli(0, x)) == 1 / sqrt(convert(GiacExpr, 3))
        @test occursin("integrate", string(∫(x * SF.besselj(0, x))))              # diverges
        @test occursin("integrate", string(∫(exp(-x) * SF.besseli(0, x))))        # diverges
        @test ∫(x * SF.besselj(0, x), 0, 1) == SF.besselj(1, convert(GiacExpr, 1))
        @test ∫(SF.besselj(1, x), 0, 2) == 1 - SF.besselj(0, two)
    end

    @testset "Bessel's and Airy's equations" begin
        @giac_var y(x)
        d(f, k = 1) = Giac.Commands.diff(f, x, k)
        for (L, head) in ((Y -> x^2 * d(Y, 2) + x * d(Y) + (x^2 - 4) * Y, "besselj"),
                          (Y -> x^2 * d(Y, 2) + x * d(Y) + (9x^2 - n^2) * Y, "bessely"),
                          (Y -> x^2 * d(Y, 2) + x * d(Y) - (x^2 + 1) * Y, "besselk"),
                          (Y -> x * d(Y, 2) + d(Y) + x * Y, "besselj"),
                          (Y -> d(Y, 2) - x * Y, "airyai"),
                          (Y -> d(Y, 2) + 4x * Y, "airybi"))
            sol = Giac.Commands.desolve(L(y) ~ 0, y)
            @test occursin(head, string(sol))
            @test iszero(Giac.invoke_cmd(:simplify, L(sol)))
        end
        @test string(Giac.Commands.desolve(x^2 * d(y, 2) + x * d(y) + (9x^2 - n^2) * y ~ 0, y)) ==
              "c_0*besselj(n,3*x)+c_1*bessely(n,3*x)"
    end

    @testset "substitution and series" begin
        z = convert(GiacExpr, 0)
        @test substitute(SF.besselj(0, x), x => 0) == 1
        @test substitute(SF.besselj(n + 1 // 2, x), n => 0) == sqrt(2 / (π_ * x)) * sin(x)
        @test substitute(SF.besselj(0, x), x => 2.0) ≈ SF.besselj(0, 2.0)
        Γ(p) = Giac.invoke_cmd(:Gamma, p)
        @test SF.airyai(z) == 1 / (convert(GiacExpr, 3)^(2 // 3) * Γ(2 // 3))
        @test to_julia(SF.airybiprime(z)) ≈ SF.airybiprime(0.0)
        S(f, a, k) = Giac.Commands.series(f, x, a, k)
        order(h, k) = h^(k + 1) * Giac.invoke_cmd(:order_size, h)
        @test S(SF.besselj(0, x), 0, 4) == 1 - x^2 / 4 + x^4 / 64 + order(x, 5)
        @test S(SF.besseli(1, 2x), 0, 3) == x + x^3 / 2 + order(x, 4)
        @test S(SF.airyai(x), 0, 2) == SF.airyai(z) + SF.airyaiprime(z) * x + order(x, 2)
        o = convert(GiacExpr, 1)
        @test S(SF.besselj(1, x), 1, 1) == SF.besselj(1, o) + (SF.besselj(0, o) - SF.besselj(2, o)) / 2 * (x - 1) + order(x - 1, 1)
        # at a zero of the argument: logs for Y and K, fractional powers, Struve functions
        val(e, v) = to_julia(Giac.invoke_cmd(:evalf, substitute(e, x => v)))
        polynomial(r) = GiacExpr(Giac._replace_nodes(m -> m isa Giac.CallNode && m.f == "order_size" ? Giac._ZERO : nothing, Giac._node(r)))
        near(f, k, v) = abs(val(polynomial(S(f, 0, k)) - f, v))
        for (f, k) in ((SF.besselk(0, x), 2), (SF.bessely(1, x), 2), (SF.besselj(1 // 3, x), 2), (Giac.struveh(1, x), 4),
                       (SF.besselk(2, x), 2), (SF.bessely(0, x) / SF.besselj(0, x), 2), (SF.besselk(3 // 4, x), 1))
            v = 0.001        # the remainder: a power past k, by a log factor at most
            @test near(f, k, v) < 10v^(k + 1 / 4) * (1 + abs(log(v))) + 1e-12 * abs(val(f, v))
        end
        @test occursin("ln(x)", string(S(SF.besselk(0, x), 0, 2)))
    end

    @testset "LaTeX" begin
        L = Giac.giac_latex
        @test L(SF.besselj(n, x)) == "J_{n}\\left(x\\right)"
        @test L(SF.hankelh2(0, x)) == "H^{(2)}_{0}\\left(x\\right)"
        @test L(SF.airyaiprime(x)) == "\\operatorname{Ai}'\\left(x\\right)"
        @test L(SF.besselj(0, x)^2) == "{J_{0}\\left(x\\right)}^{2}"
    end
end
