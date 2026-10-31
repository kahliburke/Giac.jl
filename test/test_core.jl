# The expression core: arithmetic on Julia trees, and the move between trees and GIAC objects

@testset "Core" begin
    @giac_var x y z a b
    S(e) = string(e)
    N(v) = convert(GiacExpr, v)

    @testset "sums" begin
        @test S(x + y) == "x+y" && S(y + x) == "y+x"                 # as written
        @test S(y + x + y) == "2*y+x"                                # merged where the first stood
        @test S(x - x) == "0" && S(1 + x + 2) == "3+x"
        @test S(a * x + b * x) == "a*x+b*x"                          # only numeric coefficients merge
        @test S(1.5x + x) == "2.5*x"
        @test S(x + 1 // 2) == "x+1/2"
    end

    @testset "products and powers" begin
        @test S(x * x) == "x^2" && S(x^2 * x) == "x^3" && S(x^2 / x^5) == "1/x^3"
        @test S(x / 2) == "x/2" && S(2x / 3) == "2*x/3" && S(-x / 2) == "-x/2"
        @test S(x / (y * z)) == "x/(y*z)" && S(-1 / x) == "-1/x"
        @test S((x * y)^2) == "x^2*y^2"                              # integer powers distribute
        @test S(2 * (x + 1)) == "2*(x+1)" && S((x + 1) * (x + 1)) == "(x+1)^2"   # nothing expands
        @test S(x^y * x^z) == "x^(y+z)" && S(x^(1 // 2) * x^(1 // 2)) == "x"
        @test S(sqrt(x)) == "sqrt(x)" && S(1 / sqrt(x)) == "1/sqrt(x)"
    end

    @testset "numbers" begin
        @test S(N(2)^100) == "1267650600228229401496703205376"   # no overflow
        @test S(N(typemax(Int)) + 1) == "9223372036854775808"
        @test S(sqrt(N(2))^2) == "2" && S(N(4)^(1 // 2)) == "2"
        @test S(N(im) * N(im)) == "-1"
        @test to_julia(N(3) / 4) === 3 // 4 && to_julia(x - x + 5) === 5
    end

    @testset "structure" begin
        @test isequal(x + y, y + x) && hash(x + y) == hash(y + x)
        @test Dict(x + y => 1)[y + x] == 1
        @test !isequal(N(1), N(1.0))
        @test Giac.symb_funcname(x + y) == "+" && Giac.symb_funcname(-x) == "-"
        @test Giac.symb_funcname(x^2) == "^" && Giac.symb_funcname(1 / x) == "inv"
        @test string.(Giac.arguments(x^2 - 3x)) == ["x^2", "-3*x"]
        @test giac_type(N(1 // 2)) == Giac.FRAC && giac_type(x) == Giac.IDNT && giac_type(x + 1) == Giac.SYMB
    end

    @testset "with GIAC" begin
        e = x^2 + 2x + 1
        @test S(Giac.Commands.factor(e)) == "(x+1)^2"
        @test S(substitute(e, x => 3)) == "16"
        @test S(Giac.Commands.expand((x + 1)^3) + x) == "x^3+3*x^2+4*x+1"
        @test (x + 1) == (1 + x) && sin(x)^2 + cos(x)^2 == 1
        @test S((x ~ y) + 1) == "x+1=y+1"
        @test S(lhs(x^2 ~ y)) == "x^2"
    end
end
