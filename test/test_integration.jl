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
    # convergence not decided, or a family that does not apply: left unevaluated
    @test occursin("integrate", string(C.integrate(x^b * log(x), x, 0, 1)))
    @test occursin("integrate", string(C.integrate(log(x) * log(1 + x), x, 0, 1)))
end
