# Second-order linear equations by the first two cases of Kovacic's algorithm, Riccati equations
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

    @testset "solutions of the second case" begin
        # cos √(x²+1), sin √(x²+1): GIAC gives up
        L = Y -> d(Y, 2) - d(Y) / (x * (x^2 + 1)) + x^2 / (x^2 + 1) * Y
        @test solve(L) == C0 * cos(sqrt(x^2 + 1)) + C1 * sin(sqrt(x^2 + 1))
        # e^(±√(x²+1)): GIAC's answer, e^(4√(x²+1)), fails the equation and is checked
        L = Y -> d(Y, 2) - d(Y) / (x * (x^2 + 1)) - x^2 / (x^2 + 1) * Y
        @test solve(L) == C0 * exp(sqrt(x^2 + 1)) + C1 * exp(-sqrt(x^2 + 1))
        # z'' = -(3/(16x²) + 1/(4x)) z: x^(1/4) times a cosine and a sine of √x
        E, f1, f2 = Giac._kovacic_case2(giac_eval("-3/(16*x^2) - 1/(4*x)"), x)
        @test (E, f1, f2) == (x^(1 // 4), cos(sqrt(x)), sin(sqrt(x)))
        @test Giac._kovacic_case2(convert(GiacExpr, 2) * x, x) === nothing          # Airy's: no pole
        @test Giac._fails_equation(L(y) ~ 0, exp(sqrt(x^2 + 1))^4, x, y)
        @test !Giac._fails_equation(L(y) ~ 0, C0 * exp(sqrt(x^2 + 1)), x, y)
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

    @testset "forcing terms and conditions" begin
        L = Y -> d(Y, 2) - (x^2 + 1) * Y
        f = x * exp(-x^2 / 2)
        residual_f(sol) = maximum(abs(to_julia(Giac.invoke_cmd(:evalf, substitute(L(sol) - f, Dict(x => v, C0 => 0.7, C1 => -1.3)))))
                                  for v in (0.4, 1.1))
        @test residual_f(Giac.Commands.desolve(L(y) ~ f, y)) < 1e-10                 # variation of parameters
        at(e, v) = to_julia(Giac.invoke_cmd(:evalf, substitute(e, x => v)))
        y′ = Giac.invoke_cmd(:diff, y, x)
        @test Giac.Commands.desolve([L(y) ~ 0, y(0) ~ 1, y′(0) ~ 0], y) == exp(x^2 / 2)
        s = Giac.Commands.desolve([L(y) ~ f, y(0) ~ 0, y′(0) ~ 1], y)
        @test residual_f(s) < 1e-10 && abs(at(s, 0)) < 1e-14 && at(d(s), 0) ≈ 1
        @test !occursin(")^2", string(s))                                             # e^(x²), not (e^(x²/2))²
        B = Y -> x^2 * d(Y, 2) + x * d(Y) + (x^2 - 4) * Y
        s = Giac.Commands.desolve([B(y) ~ 0, y(1) ~ 1, y′(1) ~ 0], y)
        @test at(s, 1) ≈ 1 && abs(at(d(s), 1)) < 1e-12 && abs(at(B(s), 1.7)) < 1e-12
        R = Y -> d(Y) - (Y^2 - x^2 + 1)
        s = Giac.Commands.desolve([R(y) ~ 0, y(0) ~ 1], y)
        @test at(s, 0) ≈ 1 && abs(at(R(s), 0.6)) < 1e-10
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
