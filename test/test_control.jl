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
end
