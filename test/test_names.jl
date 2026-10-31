# Names written as typed, display names, and derivatives of functions of time

@testset "Names and derivatives" begin
    T = Giac._tex_name
    L = Giac.giac_latex

    @testset "names as typed" begin
        @test T("b_0") == T("b₀") == T("b0") == "b_{0}"
        @test T("x_12") == T("x₁₂") == "x_{12}"
        @test T("x_1_2") == "x_{1,2}"
        @test T("x_ref") == "x_{\\mathrm{ref}}" && T("θ_ref") == "\\theta_{\\mathrm{ref}}"
        @test T("k_p") == T("kₚ") == "k_{p}" && T("ωₙ") == "\\omega_{n}" && T("x_θ") == "x_{\\theta}"
        @test T("a_m1") == "a_{m1}"
        @test T("V̇") == "\\dot{V}" && T("ẍ") == "\\ddot{x}" && T("x̂₁") == "\\hat{x}_{1}"
        @test T("x̂̇") == "\\dot{\\hat{x}}" && T("x⃗") == "\\vec{x}"
        @test T("Kᵀ") == "K^{\\top}" && T("x⁺") == "x^{+}" && T("P★") == "P^{\\star}"
        @test T("x′") == "x'" && T("x′★") == "x^{\\prime \\star}"
        @test T("Deltax") == "\\Delta x" && T("δu") == "\\delta u"
        @test T("xdot") == "\\mathrm{xdot}"
        @test T("dt") == "\\mathrm{d}t" && T("dq̇") == "\\mathrm{d}\\dot{q}" && T("dθ₁") == "\\mathrm{d}\\theta_{1}"
        @test T("dH") == "\\mathrm{d}H" && T("dist") == "\\mathrm{dist}"           # differentials, not words
    end

    @testset "in expressions" begin
        @giac_var Δx kₚ e₁
        @test L(Δx^2 + 1) == "\\left(\\Delta x\\right)^{2} + 1"
        @test sprint(show, Δx) == "Δx"
        @test L(kₚ * e₁) == "k_{p} e_{1}"
        cs = @giac_several_vars c -1:1
        @test L(cs[1]) == "c_{-1}"
    end

    @testset "display names" begin
        @giac_var m_c => "m_{\\text{cart}}" (kθ => "k_\\theta")::positive
        @test L(m_c) == "m_{\\text{cart}}"
        @test string(sqrt(kθ^2)) == "kθ"
    end

    @testset "functions of time and their derivatives" begin
        @giac_var t q(t) q̇ q̈ V̇ x y u
        @test string(q̇) == "diff(q(t),t)" && string(q̈) == "diff(q(t),t,2)"
        @test is_identifier(V̇)                                         # no V(t): a symbol
        @test L(q̈ + 3q̇ + 2sin(q)) == "\\ddot{q} + 3 \\dot{q} + 2 \\sin q"
        @test L(q̇^2) == "\\dot{q}^{2}"
        @giac_var ζ ωₙ
        @test L(q̈ + 2ζ * ωₙ * q̇ + ωₙ^2 * q ~ ωₙ^2 * u) ==               # ωₙ² is a parameter's square
              "\\ddot{q} + 2 \\zeta \\omega_{n} \\dot{q} + {\\omega_{n}}^{2} q = {\\omega_{n}}^{2} u"
        @test L(q(0)) == "q\\left(0\\right)"
        @test Giac.giac_latex(typeset(q̇ + q; arguments = :shown)) == "\\dot{q}\\left(t\\right) + q\\left(t\\right)"
        @test Giac.giac_latex(typeset(q̈ + q; derivatives = :prime)) == "q'' + q"
        @test Giac.giac_latex(typeset(q̇^2; derivatives = :leibniz)) == "\\left(\\frac{\\mathrm{d} q}{\\mathrm{d} t}\\right)^{2}"
        @giac_var V
        @test Giac.written_latex(:(diff(V, t))) == "\\dot{V}"            # a derivation's left side
        @test Giac.written_latex(:(diff(V, x))) == "\\frac{\\mathrm{d} V}{\\mathrm{d} x}"
        @test Giac.written_latex(:(diff(V + x, t))) == "\\frac{\\mathrm{d}}{\\mathrm{d} t}\\left(V + x\\right)"
        f = giac_eval("f(x,y)")
        @test L(Giac.Commands.diff(f, x)) == "\\frac{\\partial f}{\\partial x}"
        @test L(Giac.Commands.diff(Giac.Commands.diff(f, x), y)) == "\\frac{\\partial^{2} f}{\\partial x \\partial y}"
        @test L(Giac.Commands.diff(Giac.Commands.diff(f, x), x)) == "\\frac{\\partial^{2} f}{\\partial x^{2}}"
    end
end
