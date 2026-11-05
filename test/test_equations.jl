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

@testset "first_order" begin
    @giac_var t q(t) w(t) y(t) z(t) c k
    D = Differential(t)
    fo = first_order([D(D(q)) + c * D(q) + k * sin(q) ~ 0])
    @test string.(fo.states) == ["q", "q′"]
    @test string.(fo.rhs) == ["q′", "-c*q′-k*sin(q)"]
    @test fo.differential == 2 && string(fo.t) == "t"
    @test string(fo.derived["q′′"]) == "-c*q′-k*sin(q)"
    @test occursin("\\frac{d}{dt}", sprint(show, MIME("text/latex"), fo))

    # coupled, of different orders, in order of appearance; the accelerations solved together
    two = first_order([D(D(q)) + D(w) ~ w, D(w) - D(D(q)) ~ q])
    @test string.(two.states) == ["q", "q′", "w"]
    @test string.(Giac.Commands.simplify.(two.rhs)) == ["q′", "(-q+w)/2", "(q+w)/2"]

    # an unknown without derivatives is algebraic
    dae = first_order([D(y) ~ -y + z, z ~ y^2])
    @test string.(dae.states) == ["y", "z"] && dae.differential == 1
    @test string(dae.rhs[2]) == "z-y^2"

    @test_throws ArgumentError first_order([D(D(q))^2 ~ q])                 # not linear in q″
    @test_throws ArgumentError first_order([D(q) ~ q, D(q) ~ 2q])           # two equations for one unknown
    @test_throws ArgumentError first_order([c ~ 1])                         # no unknown functions
end
