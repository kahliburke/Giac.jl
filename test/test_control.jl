# Lie derivatives and brackets, accessibility, relative degree (Giac.Control)

using Giac.Control

@testset "Control" begin
    @giac_var θ ω px py φ x₁ x₂ x₃
    C = Giac.Commands
    same(a, b) = iszero(C.simplify(a - b))

    @testset "Lie derivatives and brackets" begin
        f = [ω, -sin(θ)]
        @test same(lie_derivative(θ^2, f, [θ, ω]), 2θ * ω)
        @test same(lie_derivative(θ^2, f, [θ, ω]; order = 2), 2ω^2 - 2θ * sin(θ))
        @test all(same.(lie_derivative([θ, ω], f, [θ, ω]), f))
        g1, g2 = [cos(φ), sin(φ), 0], [0, 0, 1]
        @test all(same.(lie_bracket(g1, g2, [px, py, φ]), [sin(φ), -cos(φ), 0]))
        @test all(same.(lie_bracket(g1, g1, [px, py, φ]), [0, 0, 0]))
        @test all(same.(ad([x₂, 0], [0, 1], [x₁, x₂], 1), [-1, 0]))
        @test all(same.(ad([x₂, 0], [0, 1], [x₁, x₂], 0), [0, 1]))
    end

    @testset "accessibility and involutivity" begin
        # the unicycle: two inputs, three states, accessible through the bracket
        gs = [[cos(φ), sin(φ), 0], [0, 0, 1]]
        @test accessibility_rank([0, 0, 0], gs, [px, py, φ]) == 3
        @test !involutive(gs, [px, py, φ])
        @test involutive([[1, 0, 0], [0, 1, 0]], [px, py, φ])
        # a double integrator, and an input that never reaches the second state
        @test accessibility_rank([x₂, 0], [0, 1], [x₁, x₂]) == 2
        @test accessibility_rank([0, 0], [1, 0], [x₁, x₂]) == 1
        # at a point
        @test accessibility_rank([0, 0], [x₁, x₂], [x₁, x₂]; at = (x₁ => 0, x₂ => 0)) == 0
    end

    @testset "relative degree and input-output linearization" begin
        @test relative_degree(x₁, [x₂, x₃, -x₁], [0, 0, 1], [x₁, x₂, x₃]) == 3
        @test relative_degree(x₃, [x₂, x₃, -x₁], [0, 0, 1], [x₁, x₂, x₃]) == 1
        @test relative_degree(x₁, [0, 0], [0, 1], [x₁, x₂]) === nothing
        io = io_linearization(θ, [ω, -sin(θ)], [0, 1], [θ, ω])
        @test io.r == 2 && string.(io.z) == ["θ", "ω"]
        @test same(io.α, sin(θ)) && same(io.β, 1)
        @test_throws ArgumentError io_linearization(x₁, [0, 0], [0, 1], [x₁, x₂])
    end

    @testset "Hurwitz stability" begin
        @giac_var s k a q::positive
        Δ = hurwitz_conditions(s^3 + 2s^2 + k * s + a, s)
        @test same(Δ[1], 2) && same(Δ[2], 2k - a) && same(Δ[3], a * (2k - a))
        @test hurwitz_stable(s^2 + q * s + 1, s) == :stable
        @test hurwitz_stable(s^2 - s + 1, s) == :unstable
        @test hurwitz_stable(s^2 + k * s + 1, s) == :unknown
        @test hurwitz_stable(s^2 + k * s + 1, s; positive = [k]) == :stable
        @test hurwitz_stable(s^3 + 2s^2 + q * s + q, s) == :stable        # Δ₂ = q, Δ₃ = q²
        @test hurwitz_stable(-s^2 - 3s - 2, s) == :stable                 # a negative leading coefficient
        # a matrix: its characteristic polynomial
        @test hurwitz_stable(GiacExpr[0 1; -2 -3], s) == :stable
        @test hurwitz_stable(GiacExpr[0 1; 2 -3], s) == :unstable
        @test_throws ArgumentError hurwitz_conditions(convert(GiacExpr, 3), s)
    end

    @testset "center manifolds" begin
        @giac_var x y z ν::positive μ
        m = center_manifold([x * y, -y + x^2], [x, y]; center = [x])
        @test same(last(only(m.h)), x^2) && same(only(m.reduced), x^3)
        @test m.stability == :unstable
        @test center_manifold([-x * y, -y + x^2], [x, y]; center = [x]).stability == :stable
        # higher order: h = x² - 4x⁴
        m5 = center_manifold([x * y, -y + x^2 - 2y^2], [x, y]; center = [x], order = 5)
        @test same(last(only(m5.h)), x^2 - 4x^4) && same(only(m5.reduced), x^3 - 4x^5)
        # two stable directions
        m2 = center_manifold([x * y + x * z, -y + x^2, -2z + x^2], [x, y, z]; center = [x])
        @test same(last(m2.h[2]), x^2 / 2) && same(only(m2.reduced), 3x^3 / 2)
        # a parameter of known sign decides; one of unknown sign does not
        @test center_manifold([-ν * x * y, -y + x^2], [x, y]; center = [x]).stability == :stable
        @test center_manifold([μ * x * y, -y + x^2], [x, y]; center = [x]).stability == :unknown
        @test occursin("asymptotically stable", sprint(show, MIME"text/plain"(),
                                                       center_manifold([-x * y, -y + x^2], [x, y]; center = [x])))
        @test_throws ArgumentError center_manifold([x + y, -y], [x, y]; center = [x])     # coupled
        @test_throws ArgumentError center_manifold([x * y, -y], [x, y]; center = [z])
    end

    @testset "weak foci: the first Lyapunov coefficient" begin
        @giac_var x₁ x₂ c::positive x y μ
        # notebook 4's pendulum with cubic damping, about the hanging position: ℓ₁ = -3c/8
        w = lyapunov_coefficient([sin(x₂) - c * x₁^3, x₁], [x₁, x₂]; at = (x₁ => 0, x₂ => π))
        @test same(w.ℓ₁, -3c / 8) && same(w.ω, 1) && w.stability == :stable
        @test lyapunov_coefficient([sin(x₂), x₁], [x₁, x₂]; at = (x₁ => 0, x₂ => π)).stability == :unknown
        # the Hopf normal form: ℓ₁ = ∓1
        @test same(lyapunov_coefficient([-y - x * (x^2 + y^2), x - y * (x^2 + y^2)], [x, y]).ℓ₁, -1)
        @test lyapunov_coefficient([-y + x * (x^2 + y^2), x + y * (x^2 + y^2)], [x, y]).stability == :unstable
        # a linear part not in normal form, ω = √2
        w2 = lyapunov_coefficient([y, -2x - x^2 * y], [x, y])
        @test same(w2.ω^2, 2) && w2.stability == :stable
        # a parameter of unknown sign
        @test lyapunov_coefficient([-y + μ * x^3, x], [x, y]).stability == :unknown
        @test occursin("supercritical", sprint(show, MIME"text/plain"(), w))
        @test_throws ArgumentError lyapunov_coefficient([-x, -y], [x, y])            # trace not zero
        @test_throws ArgumentError lyapunov_coefficient([y, x], [x, y])              # a saddle
        @test_throws ArgumentError lyapunov_coefficient([y + 1, -x], [x, y])         # not an equilibrium
    end

    @testset "center manifolds in new coordinates" begin
        @giac_var x y z μ
        # already in block form: the same variables, without center
        m = center_manifold([x * y, -y + x^2], [x, y])
        @test isempty(m.coordinates) && string(only(m.center)) == "x" && same(last(only(m.h)), x^2)
        @test string(only(center_manifold([-y + x^2, x * y], [y, x]).center)) == "x"
        # a coupled linear part, eigenvalues 0 and -2: x = ξ + η, y = ξ - η
        c = center_manifold([-x + y - x^3, x - y], [x, y])
        @test length(c.coordinates) == 2 && c.stability == :stable
        ξ = only(c.center)
        @test same(only(c.reduced), -ξ^3 / 2)
        # x + y = 2ξ on the manifold, and (x + y)' = -x³
        @test same(Giac.substitute(last(c.coordinates[1]) + last(c.coordinates[2]), first(only(c.h)) => last(only(c.h))), 2ξ)
        # a rotation and a stable direction: a two-dimensional center
        r = center_manifold([y, -x - x^2 * y, -z + x^2], [x, y, z])
        @test length(r.center) == 2 && r.stability == :unknown
        @test occursin("in coordinates with", sprint(show, MIME"text/plain"(), c))
        @test_throws ArgumentError center_manifold([y - x^3, -y + x], [x, y])     # a root in the right half-plane
        @test_throws ArgumentError center_manifold([-x, -y], [x, y])              # nothing on the axis
        @test_throws ArgumentError center_manifold([μ * x, -y], [x, y])           # parameters
    end
end
