# Typesetting as a textbook writes: conventions, and the ways to change them

@testset "Typesetting" begin
    @giac_var m g h v x a b c t τ ζ ω k s K y(t)
    L(e) = Giac.giac_latex(e)
    T(e; kw...) = Giac.giac_latex(typeset(e; kw...))
    P = convert(GiacExpr, π)

    @testset "sums" begin
        @test L(c + b * x + a * x^2) == "a x^{2} + b x + c"           # descending powers of the variable
        @test L(-x^2 + 1) == "1 - x^{2}"                             # a difference leads with its positive term
        @test L(-a + b) == "b - a"
        @test L(t * y + 3y) == "t y\\left(t\\right) + 3 y\\left(t\\right)"   # t, the argument of y(t), is a variable
        @test T(x^2 + 1 + x; order = :ascending) == "1 + x + x^{2}"
        @test T(-x^2 + 1; lead = :as_is) == "-x^{2} + 1"
        @test occursin("O\\left(", L(Giac.Commands.series(sin(x), x, 0, 5)))
        @test startswith(L(Giac.Commands.series(sin(x), x, 0, 5)), "x - ")     # a series ascends
    end

    @testset "products and fractions" begin
        @giac_var m g h
        @test L(h * g * m) == "m g h"                                 # parameters in the order declared
        @test L(x * k) == "k x"                                       # parameters before variables
        @test L(m * v^2 / 2) == "\\frac{m v^{2}}{2}"
        @test T(m * v^2 / 2; fractions = :front) == "\\frac{1}{2} m v^{2}"
        @test T(a / b; fractions = :inline) == "a/b"
        @test L(a / (b + c)) == "\\frac{a}{b + c}"
        @test L(-a / b) == "-\\frac{a}{b}"
        @test L(x^(-2)) == "\\frac{1}{x^{2}}"
        @test L(2 * (x / 3)) == "\\frac{2 x}{3}"
        @test L(2 * P) == "2 \\pi"
        @test L(a * exp(-t / τ)) == "a e^{-t/\\tau}"                  # exponents take inline fractions
        @test T(exp(x); exp = :exp) == "\\exp x"
    end

    @testset "functions and powers" begin
        @test L(sin(x)^2 + 3cos(x)^2) == "\\sin^{2} x + 3 \\cos^{2} x"
        @test T(sin(x)^2; trig_powers = :parenthesized) == "\\left(\\sin x\\right)^{2}"
        @test L(sqrt(x + 1)) == "\\sqrt{x + 1}"
        @test T(sqrt(x); roots = :power) == "x^{1/2}"
        @test L(abs(x)) == "\\left|x\\right|"
        @test L(m * x + k ~ 0) == "m x + k = 0"
    end

    @testset "choosing the variables" begin
        G = K * ω^2 / (s^2 + 2ζ * ω * s + ω^2)
        @test T(G; by = s) == "\\frac{K \\omega^{2}}{s^{2} + 2 \\zeta \\omega s + \\omega^{2}}"
        @test sprint(show, MIME("text/latex"), typeset(G; by = s)) == "\$\$" * T(G; by = s) * "\$\$"
    end

    @testset "global and scoped styles" begin
        old = typeset!()
        try
            typeset!(fractions = :front)
            @test L(m * v^2 / 2) == "\\frac{1}{2} m v^{2}"
            @test typeset(fractions = :stacked) do
                L(m * v^2 / 2)
            end == "\\frac{m v^{2}}{2}"
            @test L(m * v^2 / 2) == "\\frac{1}{2} m v^{2}"               # restored after the block
        finally
            typeset!(reset = true)
        end
        @test L(m * v^2 / 2) == "\\frac{m v^{2}}{2}"
        @test_throws ArgumentError typeset!(fractions = :sideways)
        @test_throws ArgumentError typeset!(colour = :blue)
        f = typeset(by = s)                                          # a function, for a derivation
        @test f(s + 1) isa Giac.Typeset
    end

    @testset "a style in one task does not leak into another" begin
        global_style = Giac._current_style()
        results = fetch.([Threads.@spawn(typeset(() -> L(m * v^2 / 2); fractions = (isodd(i) ? :front : :stacked)))
                          for i in 1:40])
        @test all(results[i] == (isodd(i) ? "\\frac{1}{2} m v^{2}" : "\\frac{m v^{2}}{2}") for i in 1:40)
        fetch.([Threads.@spawn(Giac.written_latex(:(x + 1))) for _ in 1:40])
        @test Giac._current_style() === global_style                 # the global style, untouched
    end

    @testset "in derivations and matrices" begin
        d = @derivation typeset(fractions = :front) begin
            E = m * v^2 / 2
        end
        @test occursin("\\frac{1}{2} m v^{2}", last(only(d.rows)))
        @test occursin("1 - x^{2}", L([1 - x^2 x; a b]))
    end
end
