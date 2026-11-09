# Second-order linear equations by the first case of Kovacic's algorithm, Riccati equations
# through their linear equations, and erfi for the erf of an imaginary argument

import SpecialFunctions as SF

@testset "Kovacic's algorithm and Riccati equations" begin
    @giac_var x y(x)
    d(f, k = 1) = Giac.Commands.diff(f, x, k)
    solve(L) = Giac.Commands.desolve(L(y) ~ 0, y)
    C0, C1 = GiacExpr(Giac.SymNode("c_0")), GiacExpr(Giac.SymNode("c_1"))
    # the residual of the equation at a few points, with values for the constants
    residual(L, sol) = maximum(abs(to_julia(Giac.invoke_cmd(:evalf, substitute(L(sol), Dict(x => v, C0 => 0.7, C1 => -1.3)))))
                               for v in (0.4, 1.1))

    @testset "exponential solutions" begin
        for (L, part) in ((Y -> d(Y, 2) - (x^2 + 1) * Y, "exp(x^2/2)"),        # GIAC gives up on these
                          (Y -> d(Y, 2) - 2x * d(Y) - 2Y, "exp(x^2)"),
                          (Y -> d(Y, 2) + x * d(Y) + Y, "exp(-x^2/2)"),
                          (Y -> d(Y, 2) - (4x^2 - 2) * Y, "exp(-x^2)"),
                          (Y -> d(Y, 2) - (1 / x^4 + 2 / x^3) * Y, "exp(1/x)"),     # a pole of order 4
                          (Y -> d(Y, 2) - (x^4 + 2x) * Y, "exp(x^3/3)"))
            sol = solve(L)
            @test occursin(part, string(sol))
            @test residual(L, sol) < 1e-10
        end
        r2 = sqrt(convert(GiacExpr, 2))
        @test solve(Y -> d(Y, 2) + x * d(Y) + Y) ==
              C0 * exp(-x^2 / 2) + C1 * sqrt(convert(GiacExpr, π)) * SF.erfi(x / r2) * exp(-x^2 / 2) / r2
        @test Giac._kovacic_case1(convert(GiacExpr, 2) * x, x) === nothing          # Airy's: no such solution
        @test !occursin("rootof", string(solve(Y -> d(Y, 2) - (x^4 + 2x) * Y)))
        # with parameters, for which the transformation's square roots keep their sign
        @giac_var a k
        @test solve(Y -> d(Y, 2) - (a^2 * x^2 + a) * Y) ==
              C0 * exp(a * x^2 / 2) + C1 * exp(a * x^2 / 2) * sqrt(convert(GiacExpr, π)) * SF.erf(sqrt(a) * x) / (2sqrt(a))
        @test occursin("exp(k*x^3/3)", string(solve(Y -> d(Y, 2) - (k^2 * x^4 + 2k * x) * Y)))
        @test Giac._gave_up(solve(Y -> d(Y, 2) - (x^2 + a) * Y))                    # parabolic cylinder
    end

    @testset "Riccati equations" begin
        L = Y -> d(Y) - (Y^2 - x^2 + 1)
        sol = solve(L)
        @test residual(L, sol) < 1e-10
        @test substitute(sol, C0 => 0) == x
        A = Y -> d(Y) - (x + Y^2)                                                    # through Airy's equation
        sol = solve(A)
        @test occursin("airyaiprime", string(sol)) && residual(A, sol) < 1e-10
    end

    @testset "erfi" begin
        π_ = convert(GiacExpr, π)
        @test Giac.Commands.integrate(exp(x^2), x) == sqrt(π_) * SF.erfi(x) / 2
        @test d(SF.erfi(x^2)) == 4x * exp(x^4) / sqrt(π_)
        @test SF.erfi(-x) == -SF.erfi(x)
        @test to_julia(SF.erfi(convert(GiacExpr, 1))) ≈ SF.erfi(1.0)
        @test Giac.giac_latex(SF.erfi(x)) == "\\operatorname{erfi}\\left(x\\right)"
    end
end
