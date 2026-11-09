# First-order linear partial differential equations by characteristics, and the heat and wave
# equations on an interval by separation of variables

@testset "Partial differential equations" begin
    @giac_var x y t
    D = Giac.Commands.diff
    π_ = convert(GiacExpr, π)
    # e with the arbitrary function F_0 made a definite one, at a point
    concrete(e) = giac_eval("eval(subst($(e), F_0 = (z_ -> sin(z_) + z_^2)))")
    at(e, p) = to_julia(Giac.invoke_cmd(:evalf, substitute(e, p)))

    @testset "first order" begin
        @giac_var u(x, y)
        for (L, form) in ((U -> D(U, x) + 2D(U, y), "F_0(-2*x+y)"),
                          (U -> D(U, x) + 2D(U, y) + U, "exp(-x)*F_0(-2*x+y)"),
                          (U -> x * D(U, x) + y * D(U, y), "F_0(y/x)"),
                          (U -> y * D(U, x) - x * D(U, y), "F_0(-x^2/2-y^2/2)"),     # circles
                          (U -> D(U, y) - 3U, "exp(3*y)*F_0(x)"))
            s = Giac.Commands.desolve(L(u) ~ 0, u)
            @test occursin("F_0", string(s))
            form === nothing || @test string(s) == form
            @test abs(at(L(concrete(s)), Dict(x => 0.3, y => 0.7))) < 1e-10
        end
        # forced, and with a condition on a line
        L = U -> D(U, x) + 2D(U, y) - x * y
        s = Giac.Commands.desolve(L(u) ~ 0, u)
        @test abs(at(L(concrete(s)), Dict(x => 0.3, y => 0.7))) < 1e-10
        s = Giac.Commands.desolve([D(u, x) + 2D(u, y) ~ 0, u(x, 0) ~ sin(x)], u)
        @test Giac.Commands.simplify(s - sin(x - y / 2)) == 0
        @test Giac.Commands.desolve([x * D(u, x) + y * D(u, y) ~ u, u(1, y) ~ y^2], u) == y^2 / x
    end

    @testset "heat and wave equations" begin
        @giac_var u(x, t)
        heat = D(u, t) ~ D(u, x, 2)
        s = Giac.Commands.desolve([heat, u(0, t) ~ 0, u(π_, t) ~ 0, u(x, 0) ~ sin(x)], u)
        @test s == exp(-t) * sin(x)
        s = Giac.Commands.desolve([D(u, t) ~ 3D(u, x, 2), u(0, t) ~ 0, u(1, t) ~ 0, u(x, 0) ~ sin(2π_ * x) - 4sin(5π_ * x)], u)
        @test Giac.Commands.simplify(s - exp(-12π_^2 * t) * sin(2π_ * x) + 4exp(-75π_^2 * t) * sin(5π_ * x)) == 0
        # insulated ends: the mean stays
        ux = Giac.invoke_cmd(:diff, u, x)
        s = Giac.Commands.desolve([heat, ux(0, t) ~ 0, ux(π_, t) ~ 0, u(x, 0) ~ 1 + cos(2x)], u)
        @test Giac.Commands.simplify(s - 1 - exp(-4t) * cos(2x)) == 0
        # an initial value with an infinite expansion: a held sum, whose terms solve the problem
        s = Giac.Commands.desolve([heat, u(0, t) ~ 0, u(π_, t) ~ 0, u(x, 0) ~ x * (π_ - x)], u)
        term, n = Giac._node(s).args[1:2]
        term, n = GiacExpr(term), GiacExpr(n)
        @test Giac.Commands.simplify(D(term, t) - D(term, x, 2)) == 0
        @test abs(sum(at(term, Dict(n => m, x => 1, t => 0)) for m in 1:400) - (π - 1)) < 1e-5
        # the wave equation, plucked and struck
        ut = Giac.invoke_cmd(:diff, u, t)
        s = Giac.Commands.desolve([D(u, t, 2) ~ 4D(u, x, 2), u(0, t) ~ 0, u(π_, t) ~ 0, u(x, 0) ~ sin(x), ut(x, 0) ~ sin(3x)], u)
        @test Giac.Commands.simplify(s - cos(2t) * sin(x) - sin(6t) * sin(3x) / 6) == 0
    end
end
