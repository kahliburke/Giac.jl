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
        # a number times a sum, in a sum, is distributed so like terms merge
        @test S(x + y - (x + y) / 2) == "x/2+y/2" && S(a - (b - x)) == "a-b+x"
        @test S(2 * (x + y)) == "2*(x+y)"                            # alone, kept
        @test S(-(x - y)) == "-x+y"                                  # a negated sum, as GIAC writes it
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
        # an integer's root takes in the powers of the integer in its coefficient
        r3 = sqrt(N(3))
        @test S(r3 - 3 / r3) == "0" && S(6 / r3) == "2*sqrt(3)" && S(r3 / 3) == "1/sqrt(3)"
        @test S(N(3)^(3 // 2)) == "3*sqrt(3)" && S(N(3)^(-3 // 2)) == "1/(3*sqrt(3))"
        @test S(x * r3 / 3 + x / r3) == "2*x/sqrt(3)" && S(r3 / 3 - 1 / r3) == "0"
        @test S(-1 / sqrt(N(2)) - 3 / sqrt(N(2))) == "-2*sqrt(2)"           # like terms merged, then folded
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

    @testset "an assumption outlives the expression it was made with" begin
        names = ["core_gc_$i" for i in 1:4]
        for n in names
            let v = GiacExpr(Giac.SymNode(n))
                Giac.Commands.assume(v > 0)
            end
        end
        foreach(_ -> GC.gc(true), 1:3)
        Giac.with_giac_lock(Giac._free_doomed!)
        @test all(n -> S(giac_eval("about($n)")) == "assume[[],[line[0,+infinity]],[0]]", names)
        foreach(n -> giac_eval("purge($n)"), names)
    end
end

# decimals at any precision count as decimals, so exact routes know to take over from them
@testset "decimals in the tree" begin
    @test Giac._has_float(Giac._node(giac_eval("evalf(sqrt(2))")))
    @test Giac._has_float(Giac._node(giac_eval("evalf(sqrt(2), 30)")))
    @test Giac._has_float(Giac._node(giac_eval("evalf(1 + i*sqrt(2), 30)")))
    @test !Giac._has_float(Giac._node(giac_eval("1/2 + 3*i")))
    @test !Giac._has_float(Giac._node(giac_eval("[1, 2]")))
end
