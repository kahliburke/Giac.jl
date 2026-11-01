# positivity: coefficient signs after writing orderings as positive gaps, checked in Nemo

using Nemo

@testset "positivity" begin
    @giac_var u t₁ t₂ t₃ l″ A A′ A₃ x y τ l′ u′ g s

    @testset "an ordering of the variables" begin
        Mt1 = (t₃ - t₁) * (A′ + A₃) * (A * (t₂ - u) * (l″ - t₁) + A′ * (l″ - t₂) * (t₁ - u))
        Mt3 = (t₃ - t₁) * (A + A′) * (A₃ * (t₂ - u) * (l″ - t₃) + A′ * (l″ - t₂) * (t₃ - u))
        for M in (Mt1, Mt3)
            r = positivity(M; order = (u, t₁, t₂, t₃, l″), positive = (A, A′, A₃))
            @test Bool(r) && r.sign == 1 && isempty(r.numerator.free)
        end
        @test positivity(t₁ - t₂; order = (t₁, t₂)).sign == -1
        @test !Bool(positivity(t₂ - t₁; order = (t₁, t₂), strict = false))     # ≥ 0 only
    end

    @testset "a rational function, and two chains" begin
        D₀ = A * (t₃ - t₂) + A′ * (t₃ - t₁) + A₃ * (t₂ - t₁)
        r = positivity((t₃ - t₂) * (t₂ - t₁) * (A₃ - A) / D₀; order = [(t₁, t₂, t₃), (0, A, A₃)], positive = (A′,))
        @test Bool(r) && r.numerator.sign == 1 && r.denominator.sign == 1
        @test length(r.gaps) == 3
    end

    @testset "a relation, by a parametrization" begin
        # t₂² = l′u′ as l′ = t₂/g, u′ = t₂g with g > 1; the bracket is -δ₊(τ - u/g)
        δ₊, δ₋, d = u′ - t₂, t₂ - l′, u′ - u
        bracket = (t₂ - τ) * δ₊ - d * δ₋
        param = (l′ => t₂ / g, u′ => t₂ * g, τ => u / g + s)
        @test Bool(positivity(-bracket; param, order = [(1, g), (0, u, t₂)], positive = (s,)))
    end

    @testset "not decided, with the terms against" begin
        r = positivity(x^2 - 2x * y + y^2; positive = (x, y))
        @test r.sign == 0 && string.(r.numerator.against) == ["-2*x*y"]
        @test occursin("sum-of-squares", sprint(show, MIME("text/plain"), r))
        @test r.numerator.terms == (2, 1) && r.numerator.degrees == 2:2
        @test r.numerator.opposing == ["x" => 1, "y" => 1]
        # a long report stays short when shown, and groups by the variables chosen
        big = positivity((x - y)^2 * (A + A′)^3 * τ^2; positive = (x, y, A, A′, τ))
        shown = sprint(show, MIME("text/plain"), big)
        @test count(==('\n'), shown) <= 8 && occursin("terms against, in:", shown)
        groups = Giac.group_terms(big.numerator.against, (A, A′))
        @test length(groups) == 4 && all(g -> string(last(g)) == "-2*x*y*τ^2" || occursin("x*y", string(last(g))), groups)
        r2 = positivity(x * τ; positive = (x,))
        @test r2.sign == 0 && string.(r2.numerator.free) == ["τ"]                    # τ's sign is not known
    end
end
