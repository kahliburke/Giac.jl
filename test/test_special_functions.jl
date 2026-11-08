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
        @test occursin("integrate", string(∫(x^2 * SF.besselj(0, x))))        # a Struve function
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

    @testset "LaTeX" begin
        L = Giac.giac_latex
        @test L(SF.besselj(n, x)) == "J_{n}\\left(x\\right)"
        @test L(SF.hankelh2(0, x)) == "H^{(2)}_{0}\\left(x\\right)"
        @test L(SF.airyaiprime(x)) == "\\operatorname{Ai}'\\left(x\\right)"
        @test L(SF.besselj(0, x)^2) == "{J_{0}\\left(x\\right)}^{2}"
    end
end
