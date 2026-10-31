# Held integrals and evaluations, and integration by parts

@testset "Held forms" begin
    L = Giac.giac_latex
    @giac_var t t₁ t₂ a η(t) η̇ p(t) F(x, v, s) q(t) q̇

    @testset "held and typeset" begin
        I = integral(F(q, q̇, t), t, t₁, t₂)
        @test L(I) == "\\int_{t_{1}}^{t_{2}} F\\left(q, \\dot{q}, t\\right) \\, \\mathrm{d}t"
        @test L(integral(a * η + p * η̇, t)) == "\\int \\left(p \\dot{\\eta} + a \\eta\\right) \\, \\mathrm{d}t"
        @test L(integral(-a * η, t, t₁, t₂)) == "-\\int_{t_{1}}^{t_{2}} a \\eta \\, \\mathrm{d}t"      # the sign outside
        @test Giac.written_latex(:(integrate(p * η̇, t, t₁, t₂))) == "\\int_{t_{1}}^{t_{2}} p \\dot{\\eta} \\, \\mathrm{d}t"
    end

    @testset "worked on inside" begin
        I = integral(a * η, t, t₁, t₂)
        @test L(substitute(p(t₂) + I, a => 2)) == "p\\left(t_{2}\\right) + \\int_{t_{1}}^{t_{2}} 2 \\eta \\, \\mathrm{d}t"
        @test L(Giac.Commands.factor(integral(a * η - p * η, t, t₁, t₂))) ==
              "\\int_{t_{1}}^{t_{2}} \\eta \\left(a - p\\right) \\, \\mathrm{d}t"
    end

    @testset "evaluations" begin
        b = evaluated(p * η, t, t₁, t₂)
        @test L(b) == "\\left. p \\eta \\right|_{t_{1}}^{t_{2}}"
        @test string(Giac.invoke_cmd(:eval, b)) == "p(t₂)*η(t₂)-p(t₁)*η(t₁)"         # GIAC's preval
        @test L(evaluated(cos(a * t), a => 0)) == "\\left. \\cos\\left(a t\\right) \\right|_{a = 0}"
        @test Giac.written_latex(:(evaluated(diff(V, ε), ε => 0))) == "\\left. \\frac{\\mathrm{d} V}{\\mathrm{d} \\varepsilon} \\right|_{\\varepsilon = 0}"
        # a substitution inside keeps the bar; one that changes its value takes the value
        @test L(substitute(evaluated(a * η, t, t₁, t₂), a => 2)) == "\\left. 2 \\eta \\right|_{t_{1}}^{t_{2}}"
        @test string(substitute(b, η(t₁) => 0)) == "p(t₂)*η(t₂)"
    end

    @testset "by parts" begin
        r = by_parts(integral(p * η̇, t, t₁, t₂), p)
        @test L(r) == "\\left. p \\eta \\right|_{t_{1}}^{t_{2}} - \\int_{t_{1}}^{t_{2}} \\dot{p} \\eta \\, \\mathrm{d}t"
        # the terms without the factor stay under the integral
        r2 = by_parts(integral(a * η + p * η̇, t, t₁, t₂), p)
        @test L(substitute(r2, η(t₁) => 0, η(t₂) => 0)) == "\\int_{t_{1}}^{t_{2}} \\left(a \\eta - \\dot{p} \\eta\\right) \\, \\mathrm{d}t"
        @test_throws ArgumentError by_parts(p * η̇, p)
        @test string(integrand(integral(-a * η, t, t₁, t₂))) == "-a*η(t)"
    end

    @testset "a function of one variable at a point" begin
        @test string(η(t₂)) == "η(t₂)"
    end
end
