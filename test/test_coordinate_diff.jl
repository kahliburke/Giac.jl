# Derivatives in a function of time and its derivatives, as independent variables

@testset "Derivatives in coordinates and velocities" begin
    L = Giac.giac_latex
    @giac_var t m g ℓ q(t) q̇ θ(t) θ̇ M(x) U(x) F(x, v, s) ε η(t) h

    @testset "q and q̇ independent" begin
        @test string(diff(q̇^2 * cos(q), q)) == "-diff(q(t),t)^2*sin(q(t))"
        @test string(diff(q̇^2 * cos(q), q̇)) == "2*diff(q(t),t)*cos(q(t))"
        @test string(diff(m * ℓ^2 * θ̇^2 / 2 + m * g * ℓ * cos(θ), θ)) == "-m*g*ℓ*sin(θ(t))"
        @test string(diff(θ̇^2 * q + sin(θ) * q^2, q)) == "diff(θ(t),t)^2+2*sin(θ(t))*q(t)"
        @test string(diff(q̇, q)) == "0"
        @test string(diff(cos(θ), t)) == "-diff(θ(t),t)*sin(θ(t))"            # other variables: the chain rule
    end

    @testset "unknown functions of the coordinates" begin
        @test string(diff(M(q) * q̇^2 / 2 - U(q), q)) == "diff(M)(q(t))*diff(q(t),t)^2/2-diff(U)(q(t))"
        @test string(diff(M(q) * q̇^2 / 2, q̇)) == "M(q(t))*diff(q(t),t)"
        Lq = F(q, q̇, t)
        @test string(diff(Lq, q̇)) == "diff(F,1)(q(t),diff(q(t),t),t)"
        @test string(diff(Lq, q)) == "diff(F,0)(q(t),diff(q(t),t),t)"
    end

    @testset "typeset" begin
        Lq = F(q, q̇, t)
        @test L(diff(Lq, q̇)) == "\\frac{\\partial F}{\\partial \\dot{q}}"
        @test L(diff(diff(Lq, q̇), t)) ==
              "\\frac{\\partial^{2} F}{\\partial \\dot{q} \\partial q} \\dot{q} + \\frac{\\partial^{2} F}{\\partial \\dot{q}^{2}} \\ddot{q} + \\frac{\\partial^{2} F}{\\partial \\dot{q} \\partial t}"
        @test L(diff(M(q) * q̇^2 / 2 - U(q), q)) == "\\frac{1}{2} \\frac{\\mathrm{d} M}{\\mathrm{d} q} \\dot{q}^{2} - \\frac{\\mathrm{d} U}{\\mathrm{d} q}"
        @test L(Giac.Commands.normal(diff(M(q), t))) == "\\frac{\\mathrm{d} M}{\\mathrm{d} q} \\dot{q}"   # function_diff(M)
        @test L(diff(M(q + h), h)) == "\\frac{\\mathrm{d}}{\\mathrm{d} h}\\left(M\\left(q + h\\right)\\right)"
        @test L(substitute(diff(M(q), t), q => q + h)) == "M'\\left(q + h\\right) \\dot{q}"
    end

    @testset "a variation of the path" begin
        Lq = F(q, q̇, t)
        Lε = substitute(Lq, q => q + ε * η)                 # q̇ varies with q
        @test L(Lε) == "F\\left(q + \\varepsilon \\eta, \\dot{q} + \\varepsilon \\dot{\\eta}, t\\right)"
        δL = substitute(diff(Lε, ε), ε => 0)
        @test L(δL) == "\\frac{\\partial F}{\\partial q} \\eta + \\frac{\\partial F}{\\partial \\dot{q}} \\dot{\\eta}"
    end

    @testset "differentials" begin
        @giac_var p ṗ x y
        Lq = F(x, y, t)
        dH = differential(p * y - Lq, x, y, p, t)
        @test string(differential(p)) == "dp" && string(differential(q̇)) == "dq̇"
        @test string(diff(dH, differential(p))) == "y"
        # in the order the variables were given, each differential after its coefficient
        @test L(dH) == "-\\frac{\\partial F}{\\partial x} \\mathrm{d}x + \\left(p - \\frac{\\partial F}{\\partial y}\\right) \\mathrm{d}y + y \\mathrm{d}p - \\frac{\\partial F}{\\partial t} \\mathrm{d}t"
        @test L(differential(q^2 * q̇, q, q̇)) == "2 q \\dot{q} \\mathrm{d}q + q^{2} \\mathrm{d}\\dot{q}"
        @test L(differential(p^2 / (2m), p)) == "\\frac{p}{m} \\mathrm{d}p"           # after the fraction
    end

    @testset "explicit dependence on time" begin
        @giac_var k Ω
        Ld = m * q̇^2 / 2 - k * q^2 / 2 + g * q * cos(Ω * t)
        @test string(partial(Ld, t)) == "-g*q(t)*Ω*sin(Ω*t)"
        @test string(partial(Ld, q̇)) == string(diff(Ld, q̇))
        @test L(partial(F(q, q̇, t), t)) == "\\frac{\\partial F}{\\partial t}"
        @test L(differential(F(q, q̇, t), q, q̇, t)) ==
              "\\frac{\\partial F}{\\partial q} \\mathrm{d}q + \\frac{\\partial F}{\\partial \\dot{q}} \\mathrm{d}\\dot{q} + \\frac{\\partial F}{\\partial t} \\mathrm{d}t"
    end

    @testset "a velocity at a point" begin
        @giac_var ω::positive q₀ v₀ q̈
        @test string(q̇(0)) == "q'(0)" && string(q̈(1)) == "q''(1)"
        @test string(Giac.Commands.desolve([q̈ + ω^2 * q ~ 0, q(0) ~ q₀, q̇(0) ~ v₀], q)) == "q₀*cos(ω*t)+v₀/ω*sin(ω*t)"
    end

    @testset "held integrals in a derivation" begin
        δL = substitute(diff(substitute(F(q, q̇, t), q => q + ε * η), ε), ε => 0)
        d = @derivation begin
            δS = integral(δL, t, 0, h)
        end
        @test d.rows[1][2] == "\\int_{0}^{h} \\left(\\frac{\\partial F}{\\partial q} \\eta + \\frac{\\partial F}{\\partial \\dot{q}} \\dot{\\eta}\\right) \\, \\mathrm{d}t"
        @test L(diff(substitute(F(q, q̇, t), q => q + ε * η), ε)) ==
              "\\partial_{1} F\\left(q + \\varepsilon \\eta, \\dot{q} + \\varepsilon \\dot{\\eta}, t\\right) \\eta + \\partial_{2} F\\left(q + \\varepsilon \\eta, \\dot{q} + \\varepsilon \\dot{\\eta}, t\\right) \\dot{\\eta}"
    end

    @testset "a function of several arguments rebuilt from its tree" begin
        n = Giac._node(F(q, q̇, t))
        @test string(Giac.invoke_cmd(:eval, GiacExpr(Giac.CallNode(n.f, n.args)))) == "F(q(t),diff(q(t),t),t)"
    end

    @testset "a function of the coordinates, declared" begin
        @giac_var G(q, q̇, t) u(t)
        @test string(G) == "G(q(t),diff(q(t),t),t)"
        @test string(u) == "u(t)"                                     # t is a symbol: as before
        @test L(diff(G, q̇)) == "\\frac{\\partial G}{\\partial \\dot{q}}"
    end

    @testset "written derivatives in a derivation" begin
        @giac_var k::positive
        Lq = m * q̇^2 / 2 - k * q^2 / 2
        V = k * q^2 / 2
        d = @derivation begin
            diff(diff(Lq, q̇), t) - diff(Lq, q) => EL
            diff(V, q)
        end
        @test d.rows[1][1] == "\\frac{\\mathrm{d}}{\\mathrm{d} t}\\left(\\frac{\\partial \\mathrm{Lq}}{\\partial \\dot{q}}\\right) - \\frac{\\partial \\mathrm{Lq}}{\\partial q}"
        @test string(EL) == string(m * diff(q, t, 2) + k * q)
        @test d.rows[2][1] == "\\frac{\\mathrm{d} V}{\\mathrm{d} q}"   # V depends on q alone
    end
end
