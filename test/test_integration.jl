# Indefinite integrals of exact derivatives of expressions in unknown functions, where GIAC's own
# integrate returns infinity

@testset "Integrating exact derivatives" begin
    @giac_var t η(t) p(t)
    C = Giac.Commands
    D(f) = C.diff(f, t)
    @test string(C.integrate(D(η * p), t)) == "p(t)*η(t)"
    @test string(C.integrate(D(η) * t + η, t)) == "t*η(t)"
    @test string(C.integrate(D(p) * sin(t) + p * cos(t), t)) == "sin(t)*p(t)"
    @test string(C.integrate(D(η^2 * p), t)) == "p(t)*η(t)^2"
    @test string(C.integrate(D(η * p) + t^2, t)) == "p(t)*η(t)+t^3/3"   # the rest to GIAC
    @test string(C.integrate(D(η) * p, t)) == "integrate(diff(η(t),t)*p(t),t)"   # no closed form
    @test string(C.integrate(t^2, t)) == "t^3/3"
end

# Definite integrals GIAC leaves unevaluated, as derivatives of Beta and Gamma integrals in their
# parameters
@testset "Integrals as derivatives in parameters" begin
    @giac_var x a::positive b
    C = Giac.Commands
    ∞ = giac_eval("+infinity")
    # the closed form, checked against the expected value
    same(f, lo, hi, v) = string(C.simplify(C.integrate(f, x, lo, hi) - giac_eval(v))) == "0"
    @test same(log(x) * log(1 - x), 0, 1, "2 - pi^2/6")
    @test same(log(x)^2 * log(1 - x), 0, 1, "2*Zeta(3) + pi^2/3 - 6")
    @test same(log(1 - x) / x, 0, 1, "-pi^2/6")                # at a pole of B(s+1, r+1): a limit
    # (x−1) as −(1−x)
    @test string(C.integrate((x - 1) * log(x) * log(1 - x), x, 0, 1) + C.integrate((1 - x) * log(x) * log(1 - x), x, 0, 1)) == "0"
    @test same(x^2 * exp(-3x) * log(x), 0, ∞, "1/9 - 2*euler_gamma/27 - 2*ln(3)/27")
    @test same(exp(1 - x) * log(x), 0, ∞, "-e*euler_gamma")
    @test same(sqrt(x) / (1 + x)^2, 0, ∞, "pi/2")
    @test same(log(1 + x) / x^(3 // 2), 0, ∞, "2*pi")
    @test same(log(x) / (1 + x)^2, 0, ∞, "0")
    # a symbolic exponent known to give a convergent integral
    r = C.integrate(x^a * log(x), x, 0, 1)
    @test string(C.simplify(r + 1 / (a + 1)^2)) == "0"
    # convergence not decided, or a family that does not apply: not taken. (GIAC's own integral of
    # x^b ln x with b unknown changes how it evaluates cot later in the process, so it isn't called.)
    @test Giac._parametric_integral(x^b * log(x), x, 0, 1) === nothing
    @test Giac._parametric_integral(log(x) * log(1 + x), x, 0, 1) === nothing
    # sinᵅ cosᵝ with their logarithms on [0, π/2]: half a Beta function
    π_ = giac_eval("pi")
    @test same(log(sin(x)), 0, π_ / 2, "-pi*ln(2)/2")
    @test same(log(cos(x))^2, 0, π_ / 2, "pi^3/24 + pi*ln(2)^2/2")
    @test same(sin(x)^2 * log(sin(x)), 0, π_ / 2, "pi/8 - pi*ln(2)/4")
end

@testset "Integrals with exponentials for sines, and by involutions of the interval" begin
    @giac_var x
    C = Giac.Commands
    ∞, π_ = giac_eval("+infinity"), giac_eval("pi")
    same(f, lo, hi, v) = string(C.simplify(C.integrate(f, x, lo, hi) - giac_eval(v))) == "0"
    # -∞ made in Julia reaches GIAC as -infinity, not as its unsigned infinity
    @test string(C.integrate(exp(-x^2), x, -∞, 0)) == "sqrt(pi)/2"
    @test same(exp(-x^2) * cos(2x), -∞, ∞, "sqrt(pi)/e")
    @test same(exp(-x^2) * sin(x)^2, -∞, ∞, "sqrt(pi)*(1 - exp(-1))/2")
    # x g(x) with g symmetric about the midpoint
    @test same(x * sin(x) / (1 + cos(x)^2), 0, π_, "pi^2/4")
    # Serret's integral, under (1-x)/(1+x); and under 1/x on [0, ∞)
    @test same(log(1 + x) / (1 + x^2), 0, 1, "pi*ln(2)/8")
    @test same(log(x) / (1 + x^2), 0, ∞, "0")
    # the sum under 1-x is 0 while the integral diverges
    @test Giac._reflected_integral(1 / x - 1 / (1 - x), x, 0, 1) === nothing
    @test string(C.integrate(1 / x - 1 / (1 - x), x, 0, 1)) == "undef"
end

@testset "Checking an antiderivative on the whole line" begin
    @giac_var x a
    chk(F, f) = Giac._is_antiderivative(F, f, x)
    @test chk(atan(x / sqrt(a)) / sqrt(a), 1 / (x^2 + a))                 # for either sign of a
    @test chk(x * abs(x) / 2, sqrt(x^2))
    @test !chk(x^2 / 2, sqrt(x^2))                                        # right for x > 0 only
    @test chk(log(x + sqrt(x^2 - 1)), 1 / sqrt(x^2 - 1))
    @test chk(2x^(3 // 2) / 3, sqrt(x))                                   # real for x > 0 only
    @test !chk(x^3, x^2)
    # an answer right for a > 0 only holds where that is assumed
    @test !chk(log(x + a), 1 / (x + a) * sqrt(a^2) / a)
    @test assuming(() -> chk(log(x + a), 1 / (x + a) * sqrt(a^2) / a), a > 0)
    @test assuming(() -> Giac._assumed_intervals(a), a > 1, a < 2) == [(1.0, 2.0)]
    # the points: both signs and all scales, the same on each run
    @test Giac._generic_point(3, 2) == Giac._generic_point(3, 2)
    @test any(p -> p[1] < 0, Giac._generic_point.(1:Giac._CHECK_POINTS, 1))
end

# The search for antiderivatives after GIAC's needs GIAC on a thread of its own
@testset "Substituting for an inverse function" begin
    if Giac._giac_threaded() && Giac._wrapper_interrupts()
        @giac_var x
        C = Giac.Commands
        # GIAC fails on each (one raises an internal error); with x = sin(t)/c and the like they become
        # polynomials in t times sines and cosines
        for s in ("(-pi*c^2*x^2+pi)^(1/2)*(a+b*acos(c*x))", "(-c^2*d*x^2+d)^(3/2)*(a+b*asin(c*x))^2",
                  "x^2*(pi*c^2*x^2+pi)^(3/2)*(a+b*asinh(c*x))", "(a+b*asinh(c*x))/(pi*c^2*x^2+pi)^(1/2)",
                  "asin(a*x)^(5/2)/(-a^2*c*x^2+c)^(1/2)")
            f = giac_eval(s)
            F = C.integrate(f, x)
            @test !occursin("integrate", string(F))
            @test Giac._is_antiderivative(F, f, x)
        end
        # sines and cosines made multiple angles before integrating: no half-angle tangents
        f = giac_eval("x^3*acos(a*x)^2")
        F = C.integrate(f, x)
        @test Giac._is_antiderivative(F, f, x) && !occursin("tan", string(F))
        # sinh and cosh integrated as exponentials come back as roots, not exp(asinh(c x))
        @test !occursin("exp", string(C.integrate(giac_eval("x^2*(pi*c^2*x^2+pi)^(3/2)*(a+b*asinh(c*x))"), x)))
    end
end

# Binomial differentials xᵐ (a + b xⁿ)ᵖ, which GIAC leaves unevaluated, by Chebyshev's substitutions
@testset "Chebyshev's substitutions for binomial differentials" begin
    if Giac._giac_threaded() && Giac._wrapper_interrupts()
        @giac_var x
        C = Giac.Commands
        # (m+1)/n + p an integer, (m+1)/n an integer, p an integer, and a sum of terms
        for s in ("1/(b*x^4+a)^(17/4)", "1/(-b*x^3+a)^(1/3)", "(b*x^4+a)^(5/4)/x^2", "x^(7/4)/(b*x+a)^(3/4)",
                  "x^3*(b*x^2+a)^(1/3)", "1/(x*(1+x^(1/3))^2)", "(d*x^4+c)^2/(b*x^4+a)^(17/4)")
            f = giac_eval(s)
            F = C.integrate(f, x)
            @test !occursin("integrate", string(F))
            @test Giac._is_antiderivative(F, f, x)
        end
        # none of Chebyshev's cases: not elementary
        @test Giac._binomial_substitution(giac_eval("(b*x^2+a)^(1/5)"), x) === nothing
    end
end

# A failed GIAC limit (a definite integral may take one) leaves GIAC's series flags changed, and
# later results with them; the wrapper restores the flags after each call
@testset "A failed limit leaves later results as they were" begin
    if isdefined(Giac.GiacCxxBindings, :series_flags)
        try
            giac_eval("limit(x^(b+1)/(b+1)*ln(x),x,0,1)")
        catch
        end
        @test Giac.GiacCxxBindings.series_flags() & 8 == 0
        @test acotd(cotd(giac_eval("2"))) == 2
    end
end

@testset "evalf to some digits leaves epsilon as it was" begin
    r = giac_eval("rootof([[-3,0,-15,0,-12],[1,0,6,0,9,0,31]])")
    eps = string(giac_eval("epsilon"))
    Giac.Commands.evalf(r, 30)                       # GIAC itself leaves epsilon at 1e-30
    @test string(giac_eval("epsilon")) == eps
    @giac_var x
    @test Giac.Commands.integrate(Giac.Commands.erfc(x)^2, x, 0, convert(GiacExpr, Inf)) ==
          (2 - sqrt(convert(GiacExpr, 2))) / sqrt(convert(GiacExpr, π))
    @giac_var t y(t)
    Giac.Commands.desolve(Giac.Commands.diff(y, t, 2) + y + y^3 ~ 0, y)   # GIAC's leaves it at 1e-32
    @test string(giac_eval("epsilon")) == eps
    # numeric checks do not depend on the session's epsilon
    giac_eval("epsilon:=1e-32")
    try
        @test string(Giac.Commands.integrate(log(1 + x) / (1 + x^2), x, 0, 1)) == "pi*ln(2)/8"
        @test string(giac_eval("epsilon")) == "1e-32"
    finally
        giac_eval("epsilon:=$eps")
    end
end

# Reciprocal substitutions u = x ± 1/x, Chebyshev's third case on both sides of 0, and parts
@testset "Reciprocal substitutions and integration by parts" begin
    @giac_var x
    C = Giac.Commands
    ∞ = convert(GiacExpr, Inf)
    # d/dx F - f at points on both sides of 0
    off(F, f) = maximum(abs(to_julia(C.evalf(substitute(C.diff(F, x) - f, x => v)))) for v in (-1.61, -0.37, 0.83, 1.61))
    for f in ((x^2 - 1) / ((x^2 + 1) * sqrt(x^4 + 1)), (x^2 + 1) / ((x^2 - 1) * sqrt(x^4 + 1)))
        F = Giac._reciprocal_substitution(f, x)
        @test F !== nothing && off(F, f) < 1e-9
    end
    F = Giac._reciprocal_substitution(1 / sqrt(x^4 + 1), x)                     # elliptic, in u = x - 1/x
    @test F !== nothing && occursin("ellipf", string(F)) && off(F, 1 / sqrt(x^4 + 1)) < 1e-9
    f = 1 / (1 + x^4)^(1 // 4)
    F = Giac._binomial_substitution(f, x)
    @test F !== nothing && off(F, f) < 1e-9
    f = x * exp(-x^2) * C.erfc(x)                                                # parts, once
    F = Giac._parts_antiderivative(f, x)
    @test F !== nothing && off(F, f) < 1e-9
    F = Giac._parts_antiderivative(C.erf(x)^2, x)                                # parts, twice
    @test F !== nothing && off(F, C.erf(x)^2) < 1e-9
    @test C.integrate(C.erfc(x)^2, x, 0, ∞) == (2 - sqrt(convert(GiacExpr, 2))) / sqrt(convert(GiacExpr, π))
end

# Pseudo-elliptic integrals, from a unit U + V√P of the continued fraction of √P
@testset "Pseudo-elliptic integrals" begin
    @giac_var x
    C = Giac.Commands
    off(F, f) = maximum(abs(to_julia(C.evalf(substitute(C.diff(F, x) - f, x => v)))) for v in (-1.61, 0.83, 5.3))
    P = x^4 + 10x^2 - 96x - 71
    U = x^8 + 20x^6 - 128x^5 + 54x^4 - 1408x^3 + 3124x^2 + 10001
    V = x^6 + 15x^4 - 80x^3 + 27x^2 - 528x + 781
    F = Giac._pseudo_elliptic(x / sqrt(P), x)                                     # Chebyshev's
    @test F == log(U + V * sqrt(P)) / 8
    @test C.integrate(x / sqrt(P), x) == F
    F = Giac._pseudo_elliptic(3x / sqrt(4P), x)                                   # a square leading coefficient
    @test F !== nothing && off(F, 3x / sqrt(4P)) < 1e-9
    @test Giac._pseudo_elliptic(1 / sqrt(P), x) === nothing                       # elliptic
    @test Giac._pseudo_elliptic(x / sqrt(x^4 + 1 + x), x) === nothing
end

@testset "Factoring over the rationals" begin
    @giac_var x
    C = Giac.Commands
    F = C.factor(x^105 - 1)
    @test length(Giac._node(F).factors) == 8 && !occursin("sqrt", string(F))     # Φ_d for d | 105
    @test string(C.factor(x^4 + 1)) == "x^4+1"
    @test occursin("sqrt(2)", string(C.factor(x^4 + 1, sqrt(convert(GiacExpr, 2)))))
end

@testset "Factoring over other domains" begin
    @giac_var x y
    C = Giac.Commands
    f(p, K) = string(C.factor(p, K))
    # the product of the factors is p
    same(p, K) = all(t -> abs(Giac._approx_value(substitute(C.factor(p, K) - p, x => t))) < 1e-9, (0.37, -1.3, 2.1 + 0.4im))
    @test f(6x^2 - 12, ℤ) == "(2*3)*(x^2-2)"
    @test f(-12x^3 + 12x, ℤ) == "-(2^2*3)*x*(x-1)*(x+1)"
    @test f(x^2 - y^2, ℤ) == "(x-y)*(x+y)"
    @test_throws ArgumentError C.factor(x^2 / 2 - 1, ℤ)
    @test f(x^4 + 1, ℚ) == "x^4+1"
    @test f(x^4 + 1, Giac.GF(5)) == "((1 % 5)*x^2+2 % 5)*((1 % 5)*x^2-2 % 5)"
    @test f(x^2 / 3 + 1, Giac.GF(5)) == "(2 % 5)*((1 % 5)*x^2+(-2 % 5))"     # 1/3 is 2 mod 5
    @test_throws ArgumentError C.factor(x^2 / 5 + 1, Giac.GF(5))
    @test_throws ArgumentError Giac.GF(6)
    # ℝ: quadratics with no real roots stay; ℂ: linear factors
    @test f(x^4 - 2, ℝ) == "(x-2^(1/4))*(x+2^(1/4))*(x^2+sqrt(2))"
    @test f(x^4 + 1, ℝ) == "(x^2-sqrt(2)*x+1)*(x^2+sqrt(2)*x+1)"
    @test f((x^2 + 1)^2 * (x - 3), ℂ) == "(x-3)*(x-i)^2*(x+i)^2"
    for p in (x^4 + 1, x^4 - 2, 2x^2 + 3, x^6 - 1, 3x^2 - 5x + 1, x^8 - 16), K in (ℝ, ℂ)
        @test same(p, K)
    end
    @test !occursin("x^2", f(x^8 - 16, ℂ))
    @test f(7 * one(GiacExpr), ℝ) == "7"
    @test_throws ArgumentError C.factor(x^3 - 2, ℝ)          # roots beyond square roots
    @test_throws ArgumentError C.factor(x^2 - y^2, ℝ)
    # a complex constant in a sum is written with the signs of its parts
    I = convert(GiacExpr, im)
    @test string(x - I) == "x-i" && string(x - 1 - I) == "x-1-i" && string(x - 1 // 2 - 3I // 2) == "x-1/2-3/2*i"
end
