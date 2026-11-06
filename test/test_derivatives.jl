# Derivatives taken on the expression tree, and handed to GIAC where the tree does not model them

@testset "Tree derivatives" begin
    @giac_var x y t q(t)
    C = Giac.Commands
    S(e) = string(e)
    giac_diff(f, v) = giac_eval("diff($(S(f)), $(S(v)))")
    native(e) = getfield(e, :native)

    @testset "agree with GIAC" begin
        for f in (x^3 * sin(x), exp(2x) * cos(x^2 + y), log(x) / x, sqrt(1 + x^2), x^y, 2^x, tan(x)^2,
                  atan(x / y), asin(x), acos(2x), sinh(x) * cosh(x), tanh(x), (x + 1)^(1 // 3) / (x - 1),
                  x^2.5, exp(x) * exp(-x^2), log(sin(x)) * y, x^x)
            d = C.diff(f, x)
            @test native(d)
            @test S(C.simplify(d - giac_diff(f, x))) in ("0", "0.0")
        end
        # GIAC writes asinh, acosh, atanh as logarithms, so these are checked by value
        for (f, df) in ((asinh(x), 1 / sqrt(x^2 + 1)), (acosh(x), 1 / sqrt(x^2 - 1)), (atanh(x), 1 / (1 - x^2)))
            @test S(C.simplify(Giac._tree_diff(f, x) - df)) == "0"
        end
    end

    @testset "forms" begin
        @test S(C.diff(x^3 * sin(x), x)) == "3*x^2*sin(x)+x^3*cos(x)"
        @test S(C.diff(sqrt(1 + x^2), x)) == "x/sqrt(1+x^2)"
        @test S(C.diff(x^2 * y, y)) == "x^2" && S(C.diff(x * y, t)) == "0"
        @test S(C.diff(x^4, x, 2)) == "12*x^2" && S(C.diff(x^4, x, 0)) == "x^4"
        @test S(diff(sin(x) * y, x)) == "y*cos(x)"
    end

    @testset "left to GIAC" begin
        # an unknown function, a function the tree has no rule for, a vector, a variable that is not a symbol
        @test S(C.diff(sin(q), t)) == S(giac_diff(sin(q), t)) && !native(C.diff(sin(q), t))
        @test S(C.diff(abs(x), x)) == S(giac_diff(abs(x), x))
        @test S(C.diff(giac_eval("[x^2, y*x]"), x)) == "[2*x,y]"
        @test S(C.diff(x^2 * y, [x, y])) == S(giac_diff(x^2 * y, giac_eval("[x, y]")))
        # a part without the variable needs no rule
        @test S(C.diff(abs(y) * x^2, x)) == "2*abs(y)*x"
    end
end
