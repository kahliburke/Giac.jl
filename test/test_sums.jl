# Sums and products GIAC leaves unevaluated, hypergeometric functions, and polynomial
# decomposition

@testset "Sums, products and hypergeometric functions" begin
    @giac_var n k x a
    C = Giac.Commands
    ∞ = giac_eval("+infinity")
    sum_(args...) = string(invoke_cmd(:sum, args...))

    @testset "alternating sums, in pairs of terms" begin
        @test sum_((-1)^n / (2n + 1)^3, n, 0, ∞) == "pi^3/32"     # polygamma values reflected
        @test sum_((-1)^n / (2n + 1), n, 0, ∞) == "pi/4"
        @test sum_((-1)^n / n, n, 1, ∞) == "-ln(2)"              # each half diverges
    end

    @testset "infinite products as limits" begin
        @test string(invoke_cmd(:product, 1 - 1 / n^2, n, 2, ∞)) == "1/2"
        @test string(invoke_cmd(:product, 4n^2 / (4n^2 - 1), n, 1, ∞)) == "pi/2"
    end

    @testset "Chu–Vandermonde" begin
        @test sum_(C.binomial(n, k)^2, k, 0, n) == "(2*n)!/(n!)^2"
        @test sum_(C.binomial(n, k) * C.binomial(a, k), k, 0, n) == "(a+n)!/(a!*n!)"
        @test sum_(1 / n^4, n, 1, ∞) == "1/90*pi^4"             # GIAC's own
    end

    @testset "hypergeometric" begin
        @test string(Giac.hypergeometric([1, 1], [2], x)) == "-ln(abs(x-1))/x"
        @test string(Giac.hypergeometric([1], [1], x)) == "exp(x)"
        @test string(Giac.hypergeometric([1, 2], [1], x)) == "1/(x^2-2*x+1)"
        @test iszero(C.simplify(Giac.hypergeometric([-3, 1 // 2], [3 // 2], x) - (1 - x + 3x^2 / 5 - x^3 / 7)))
        @test_throws ArgumentError Giac.hypergeometric([1 // 3], [1 // 2], x)
    end
end

@testset "Polynomial decomposition" begin
    @giac_var x
    C = Giac.Commands
    f = x^6 + 6x^4 + x^3 + 9x^2 + 3x - 5
    g, h = Giac.polynomial_decomposition(f, x)
    @test string(g) == "x^2+x-5" && string(h) == "x^3+3*x"
    @test iszero(C.expand(substitute(g, x => h) - f))
    @test length(Giac.polynomial_decomposition(C.expand(((x^2 + x)^2 + 1)^3 - 2), x)) == 3
    @test length(Giac.polynomial_decomposition(x^5 + x + 1, x)) == 1       # prime degree
    @test length(Giac.polynomial_decomposition(x^4 + x + 1, x)) == 1
    @test_throws ArgumentError Giac.polynomial_decomposition(x^4 + sqrt(GiacExpr(2)) * x^2, x)
end
