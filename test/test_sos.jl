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
end
