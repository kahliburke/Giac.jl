# Equations as values: sides, arithmetic on both sides, isolate, substitution by equation

@testset "Equations" begin
    @giac_var x y a b
    S = string
    eq = 2x + 3 ~ y

    @testset "sides" begin
        @test S(lhs(eq)) == "2*x+3"
        @test S(rhs(eq)) == "y"
        @test S.(rhs.([x ~ 1, y ~ 2])) == ["1", "2"]
        @test_throws ArgumentError lhs(x + 1)
        @test Giac._is_equation(eq)
        @test !Giac._is_equation(x + 1)
        @test !Giac._is_equation(giac_eval("3"))
    end

    @testset "arithmetic applies to both sides" begin
        @test S(eq - 3) == S(2x ~ y - 3)
        @test S(eq + 1) == S(lhs(eq) + 1 ~ y + 1)
        @test S(2eq) == S(2 * (2x + 3) ~ 2y)
        @test S(eq / 2) == S((2x + 3) / 2 ~ y / 2)
        @test S(-eq) == S(-(2x + 3) ~ -y)
        @test S(eq^2) == S((2x + 3)^2 ~ y^2)
        @test S(eq + (a ~ b)) == S(2x + 3 + a ~ y + b)
        @test S(1 / eq) == S(1 / (2x + 3) ~ 1 / y)
        @test isequal(Giac.Commands.simplify((eq - 3) / 2), x ~ (y - 3) / 2)
        # ordinary arithmetic is unchanged
        @test S(x + 1) == "x+1"
    end

    @testset "isolate" begin
        @test S(isolate(eq, x)) == S(x ~ (y - 3) / 2)
        sols = isolate(x^2 ~ a, x)
        @test sols isa Vector{GiacExpr} && length(sols) == 2
        @test all(s -> S(lhs(s)) == "x", sols)
        @test Set(S.(isolate(x^2 - 4, x))) == Set(["x=-2", "x=2"])
        @test_throws ArgumentError isolate(x ~ x + 1, x)
    end

    @testset "substitute with equations" begin
        @test S(substitute(x + y, y ~ 2x)) == S(substitute(x + y, y => 2x))
        @test S(substitute(x * y + a, [y ~ 2x, a ~ 1])) == S(substitute(x * y + a, y => 2x, a => 1))
        @test S(substitute(x * y + a, y ~ 2x, a => 1)) == S(substitute(x * y + a, y => 2x, a => 1))
        @test S(substitute(y^2, isolate(eq, y))) == S(substitute(y^2, y => 2x + 3))
    end

    @testset "a float keeps its digits" begin
        @test S(x ~ 0.123456789012345) == "x=0.123456789012345"
    end
end
