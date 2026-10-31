# Writing an expression in the form of a template

@testset "in_form" begin
    @giac_var s K ζ ω x a h k A B b0 a0 a1
    second_order = K * ω^2 / (s^2 + 2ζ * ω * s + ω^2)

    @testset "numbers" begin
        f = in_form(5 / (s^2 + 3s + 4), second_order, (K, ζ, ω))
        @test string.(values(f)) == ["5/4", "3/4", "2"]
        @test string(f[ω]) == "2" && haskey(f, ζ) && !haskey(f, s)
        @test string(only(f.alternatives)[ω]) == "-2"                 # ω = -2 also fits
        @test string(substitute(a * (x - h)^2 + k, in_form(2x^2 - 4x + 7, a * (x - h)^2 + k, (a, h, k)))) == "2*(x-1)^2+5"
        g = in_form((3s + 5) / (s^2 + 3s + 2), A / (s + 1) + B / (s + 2), (A, B))
        @test string(substitute(A / (s + 1) + B / (s + 2), g)) == "2/(s+1)+1/(s+2)"
        @test string(f) == "K*ω^2/(s^2+2*ζ*ω*s+ω^2) with K = 5/4, ζ = 3/4, ω = 2"
    end

    @testset "symbols" begin
        f = in_form(b0 / (s^2 + a1 * s + a0), second_order, (K, ζ, ω))
        @test string.(values(f)) == ["b0/a0", "a1/(2*sqrt(a0))", "sqrt(a0)"]
        @test Giac._in_form_latex(f) == "\\frac{K \\omega^{2}}{s^{2} + 2 \\zeta \\omega s + \\omega^{2}}, " *
                                        "\\qquad K = \\frac{b_{0}}{a_{0}},\\; \\zeta = \\frac{a_{1}}{2 \\sqrt{a_{0}}},\\; \\omega = \\sqrt{a_{0}}"
    end

    @testset "declared signs and failures" begin
        @giac_var w::positive
        f = in_form(5 / (s^2 + 3s + 4), K * w^2 / (s^2 + 2ζ * w * s + w^2), (K, ζ, w))
        @test string(f[w]) == "2" && isempty(f.alternatives)
        @test_throws ArgumentError in_form(x^3, a * (x - h)^2 + k, (a, h, k))
        @test_throws ArgumentError in_form(x, a * k, (a, k))
    end
end
