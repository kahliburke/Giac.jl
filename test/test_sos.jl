# sums of squares and Lyapunov functions, certified exactly (GiacClarabelExt)

using JuMP, Clarabel

@testset "sums of squares" begin
    @giac_var x y x₁ x₂
    C = Giac.Commands
    exact(c) = iszero(C.expand(GiacExpr(c) - c.polynomial)) && all(>(0), c.weights)

    @testset "certificates" begin
        c = sos_decomposition(x^2 - 2x * y + 2y^2)
        @test c !== nothing && exact(c)
        @test iszero(C.expand(c.polynomial - (x^2 - 2x * y + 2y^2)))
        # a singular Gram matrix: the square vanishes on a line
        c = sos_decomposition((x - y)^2)
        @test c !== nothing && exact(c) && length(c.squares) == 1
        # a quartic with cross terms
        c = sos_decomposition(2x^4 + 2x^3 * y - x^2 * y^2 + 5y^4)
        @test c !== nothing && exact(c)
        c = sos_decomposition(x^4 + 2x^2 * y^2 + y^4 + 1 - x * y)
        @test c !== nothing && exact(c)
        @test occursin("Sum of squares", sprint(show, MIME"text/plain"(), c))
        @test sos_decomposition(0 * x) !== nothing
    end

    @testset "no certificate" begin
        @test sos_decomposition(x^2 - y^2) === nothing             # indefinite
        @test sos_decomposition(x^3 + 1) === nothing               # odd degree
        # Motzkin's polynomial is nonnegative but not a sum of squares
        @test sos_decomposition(x^4 * y^2 + x^2 * y^4 - 3x^2 * y^2 + 1) === nothing
        @test_throws ArgumentError sos_decomposition(1 / (x^2 + 1))
    end

    @testset "positivity decided by a sum of squares" begin
        @giac_var a b τ
        # mixed coefficient signs: nonnegative, not strictly
        r = positivity(a^2 - 2a * b + b^2; positive = (a, b))
        @test r.sign == 1 && !r.strict && r.numerator.sos !== nothing
        @test isempty(r.numerator.against)
        @test occursin("a sum of", sprint(show, MIME"text/plain"(), r))
        # a variable of unknown sign is allowed: τ² - 2τ + 2 = (τ - 1)² + 1
        @test positivity(τ^2 - 2τ + 2).sign == 1
        # Motzkin's polynomial times 1 + x² + y² is a sum of squares
        m = x^4 * y^2 + x^2 * y^4 - 3x^2 * y^2 + 1
        r = positivity(m)
        @test r.sign == 1 && r.numerator.sos !== nothing
        c = r.numerator.sos
        @test iszero(C.expand(GiacExpr(c) - c.multiplier * c.polynomial)) && string(c.multiplier) != "1"
        @test sign_of(m) === :nonnegative
        # the negative of a sum of squares
        @test positivity(-(a - b)^2; positive = (a, b)).sign == -1
        # with a, b written as squares: a⁴ - a²b² + b⁴ is a sum of squares
        @test positivity(a^2 - a * b + b^2; positive = (a, b)).sign == 1
        # indefinite: still not decided
        @test positivity(a^2 - 3a * b + b^2; positive = (a, b)).sign == 0
        @test positivity(a^2 - 2a * b + b^2; positive = (a, b), sos = false).sign == 0
    end

    @testset "Lyapunov functions" begin
        # V = x₁² + x₂² works: V̇ = -2x₁² - 2x₂²
        f = [-x₁ - x₁ * x₂^2, -x₂ + x₁^2 * x₂]
        L = lyapunov(f, [x₁, x₂])
        @test L !== nothing && L.δ > 0
        @test exact(L.positive) && exact(L.decreasing)
        sq = x₁^2 + x₂^2
        @test iszero(C.expand(L.positive.polynomial - (L.V - L.ε * sq)))
        @test iszero(C.expand(L.decreasing.polynomial - (-L.Vdot - L.δ * sq)))
        V̇ = C.diff(L.V, x₁) * f[1] + C.diff(L.V, x₂) * f[2]
        @test iszero(C.expand(V̇ - L.Vdot))
        @test occursin("global asymptotic stability", sprint(show, MIME"text/plain"(), L))
        # a linear system with a coupling: a quadratic V that is not |x|²
        @test lyapunov([-x₁ + 3x₂, -x₂], [x₁, x₂]) !== nothing
        # unstable: no Lyapunov function
        @test lyapunov([x₁, -x₂], [x₁, x₂]) === nothing
        @test_throws ArgumentError lyapunov([x₁ + 1, x₂], [x₁, x₂])
        @test_throws ArgumentError lyapunov([x₁, x₂], [x₁, x₂]; degree = 3)
    end

    @testset "regions of attraction" begin
        # ẋ = -x + x³ is attracted to 0 exactly for |x| < 1: V = x²/2 below 1/2
        r = region_of_attraction([-x + x^3], [x])
        @test r !== nothing && string(r.V) == "x^2/2"
        @test 0.49 < r.c < 1 // 2
        @test exact(r.positive) && exact(r.multiplier) && exact(r.decreasing)
        s = GiacExpr(r.multiplier)
        @test iszero(C.expand(r.decreasing.polynomial - (-r.Vdot - r.ε * x^2 - s * (r.c - r.V))))
        @test occursin("Region of attraction contains", sprint(show, MIME"text/plain"(), r))
        # reversed-time Van der Pol
        vdp = region_of_attraction([-x₂, x₁ + (x₁^2 - 1) * x₂], [x₁, x₂])
        @test vdp !== nothing && vdp.c > 0
        @test exact(vdp.decreasing)
        # an unstable linearization needs a V
        @test_throws ArgumentError region_of_attraction([x], [x])
    end
end
