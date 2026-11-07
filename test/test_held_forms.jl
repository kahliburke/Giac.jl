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
        # written `substitute` with one pair is the same bar; a written rational is a fraction
        @test Giac.written_latex(:(substitute(G, s => im * ω))) == "\\left. G \\right|_{s = i \\omega}"
        @test Giac.written_latex(:(P(t^2, 3) - 1 // 10)) == "P\\left(t^{2}, 3\\right) - \\frac{1}{10}"
        # several pairs share one bar, three pairs are not `evaluated`'s preval form, and a
        # substitution inside real/imag keeps its bar
        @test Giac.written_latex(:(substitute(G, s => 1, t => 2))) == "\\left. G \\right|_{s = 1, t = 2}"
        @test Giac.written_latex(:(substitute(G, s => 1, t => 2, u => 3))) == "\\left. G \\right|_{s = 1, t = 2, u = 3}"
        @test Giac.written_latex(:(real(substitute(G, s => im * ω)))) == "\\operatorname{Re}\\left(\\left. G \\right|_{s = i \\omega}\\right)"
        @test Giac.written_latex(:(imag(G(im * ω)))) == "\\operatorname{Im}\\left(G\\left(i \\omega\\right)\\right)"
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
        # the factor may be part of a power: x divides x² sin x
        @giac_var x
        @test string(by_parts(integral(x^2 * sin(x), x), x)) == "(sin(x)-x*cos(x))*x+integrate(x*cos(x)-sin(x),x)"
        @test string(integrand(integral(-a * η, t, t₁, t₂))) == "-a*η(t)"
    end

    @testset "twice by parts, derivatives at the limits" begin
        @giac_var x b::positive f(x) g(x)
        I = substitute(by_parts(integral(f * diff(g, x, 2), x, 0, b), f), f(0) => 0, f(b) => 0)
        @test string(I) == "-integrate(diff(f(x),x)*diff(g(x),x),x,0,b)"
        # a multiple of an integral, its sign distributed over the result
        I = by_parts(I, diff(f, x))
        @test L(I) == "\\int_{0}^{b} \\frac{\\mathrm{d}^{2} f}{\\mathrm{d} x^{2}} g\\left(x\\right) \\, \\mathrm{d}x - \\left. \\frac{\\mathrm{d} f}{\\mathrm{d} x} g\\left(x\\right) \\right|_{0}^{b}"
        @test string(substitute(I, g(0) => 0, g(b) => 0)) == "integrate(diff(f(x),x,2)*g(x),x,0,b)"
        # an integral stands after its coefficient's fraction, as a sum holding one does
        @giac_var m::positive
        J = -1 / (2m) * integral(f * g, x, 0, b)
        @test L(J) == "-\\frac{1}{2 m} \\int_{0}^{b} f\\left(x\\right) g\\left(x\\right) \\, \\mathrm{d}x"
        @test startswith(L(by_parts(J, f)), "-\\frac{1}{2 m} \\left(")
    end

    @testset "a function of one variable at a point" begin
        @test string(η(t₂)) == "η(t₂)"
    end
end
