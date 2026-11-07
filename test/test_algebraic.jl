# Algebraic numbers, decided exactly in Nemo's field of algebraic numbers

using Nemo
using Giac: to_giac, to_nemo

@testset "algebraic numbers" begin
    K = Nemo.QQBar
    @giac_var x
    N(v) = convert(GiacExpr, v)
    G(s) = giac_eval(s)
    Qt, t = polynomial_ring(QQ, "t")

    @testset "into QQBar" begin
        @test to_nemo(sqrt(N(2)) + sqrt(N(3)), K) == sqrt(K(2)) + sqrt(K(3))
        @test to_nemo(G("(2+sqrt(5))^(1/3)"), K) == (1 + sqrt(K(5))) // 2
        @test to_nemo(G("cos(pi/7)"), K) == cospi(K(1 // 7))
        @test to_nemo(G("1/2+i*sqrt(3)/2"), K) == exp_pi_i(K(1 // 3))
        @test to_nemo(G("surd(-8,3)"), K) == K(-2)
        @test to_nemo(G("(-8)^(1/3)"), K) == 2 * exp_pi_i(K(1 // 3))       # the principal branch
        # rootof(P, Q) is P at the root of Q with the largest real part, then imaginary part
        @test to_nemo(G("rootof([1,0],[1,0,-3,1])"), K) == maximum(Nemo.roots(K, t^3 - 3t + 1))
        @test to_nemo(G("rootof([1,0],[1,0,0,0,1])"), K) == exp_pi_i(K(1 // 4))
        @test to_nemo(G("rootof([1,0,0],[1,0,-3,1])"), K) == maximum(Nemo.roots(K, t^3 - 3t + 1))^2
        @test_throws ErrorException to_nemo(G("pi"), K)
        @test_throws ErrorException to_nemo(x + 1, K)
        @test_throws ErrorException to_nemo(G("cos(1)"), K)
        @test_throws ErrorException to_nemo(G("0.5"), K)
    end

    @testset "back to GIAC" begin
        @test string(to_giac(K(3) // 4)) == "3/4"
        @test string(to_giac((1 + sqrt(K(5))) // 2)) == "1/2+sqrt(5)/2"
        @test string(to_giac((1 - sqrt(K(5))) // 2)) == "1/2-sqrt(5)/2"
        @test string(to_giac(sqrt(K(-3)))) == "(i)*sqrt(3)"
        @test string(to_giac(Nemo.root(K(2), 3))) == "2^(1/3)"
        # every root of a polynomial comes back as itself, as a radical or a rootof
        for p in (t^3 - 3t + 1, t^5 - t - 1, t^4 + t^3 - 4t^2 - 4t + 1, t^3 - 2, 5t^5 - 3, t^4 + 1)
            for r in Nemo.roots(K, p)
                @test to_nemo(to_giac(r), K) == r
            end
        end
        rs = sort(Nemo.roots(K, t^3 - 3t + 1))
        @test string(to_giac(rs[3])) == "rootof([1,0],[1,0,-3,1])"
        @test string(to_giac(rs[2])) == "rootof([-1,-3,0],[1,6,9,1])"
        @test string(to_giac(rs[1])) == "rootof([-1,0],[1,0,-3,-1])"
    end

    @testset "minimal polynomials and roots" begin
        @test string(Giac.minimal_polynomial(sqrt(N(2)) + sqrt(N(3)), x)) == "x^4-10*x^2+1"
        @test string(Giac.minimal_polynomial(G("cos(pi/7)"), x)) == "8*x^3-4*x^2-4*x+1"
        r = Giac.algebraic_roots(x^3 - 3x + 1, x)
        @test length(r) == 3 && [to_nemo(e, K) for e in r] == sort(Nemo.roots(K, t^3 - 3t + 1))
        @test string.(Giac.algebraic_roots((x - 1)^2 * (x^2 + 2), x)) == ["1", "1", "(-i)*sqrt(2)", "(i)*sqrt(2)"]
        r5 = Giac.algebraic_roots(x^5 - x - 1, x)
        @test length(r5) == 5 && count(e -> isreal(to_nemo(e, K)), r5) == 1
        @test_throws ArgumentError Giac.algebraic_roots(x^2 - G("y"), x)
        # a root typesets as the root of its minimal polynomial near a decimal
        L = Giac.giac_latex
        @test L(r[2]) == "\\operatorname{root}_{\\approx 0.347}\\left(x^{3} - 3 x + 1\\right)"
        @test L(r5[5]) == "\\operatorname{root}_{\\approx 0.181 + 1.084 i}\\left(x^{5} - x - 1\\right)"
    end

    @testset "simplify_algebraic" begin
        @test string(Giac.simplify_algebraic(G("(2+sqrt(5))^(1/3)") * x)) == "(1/2+sqrt(5)/2)*x"
        @test string(Giac.simplify_algebraic(G("cos(pi/5) - sin(pi/10)"))) == "1/2"
        @test string(Giac.simplify_algebraic(G("sqrt(2)*sqrt(3) - sqrt(6)") * x + 1)) == "1"
        # values at multiples of π/7 share one field of degree 6, so their sum is decided
        @test string(Giac.simplify_algebraic(G("cos(2*pi/7) + cos(4*pi/7) + cos(6*pi/7)"))) == "-1/2"
        # a constant whose canonical form is longer is kept as written
        @test string(Giac.simplify_algebraic(sin(x) + G("cos(pi/7)"))) == string(sin(x) + G("cos(pi/7)"))
    end

    @testset "exact signs of constants" begin
        @test sign_of(sqrt(N(2)) + sqrt(N(3)) - sqrt(N(10))) == :negative
        @test sign_of(G("rootof([1,0],[1,0,-3,1])") - N(153 // 100)) == :positive
        @test sign_of(G("cos(pi/7)") - N(9 // 10)) == :positive
        @test sign_of(sqrt(N(2)) * sqrt(N(3)) - sqrt(N(6))) == :zero
        @test sign_of(x^2 + sqrt(N(2)) - 1) == :positive
        @test sign_of(-x^2 - G("cos(pi/7)")) == :negative
        @test sign_of(x^2 + sqrt(N(2)) - 3 // 2) == :unknown              # negative near x = 0
        @test sign_of(G("pi") - 3) == :unknown                              # not algebraic
    end
end
