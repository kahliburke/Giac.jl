# decompose: a target as a combination of blocks with unknown multipliers, solved exactly in Nemo

using Nemo

@testset "decompose" begin
    C = Giac.Commands
    @giac_var p r L x y

    @testset "a constant multiplier" begin
        target = (p * (L - r) + (L - p) * r)^2 - (p - r)^2 * L^2
        d = decompose(target, [:Q => p * (L - p) * r * (L - r)])
        @test string(only(d.multipliers)) == "4" && isempty(d.directions)
        @test occursin("(4) · Q", sprint(show, MIME("text/plain"), d))
    end

    @testset "a family of solutions" begin
        d = decompose(x^3 + y^3, [x + y, x * y]; degree = [2, 1])
        @test length(d.directions) == 1
        for λ in (0, 1, -3 // 2)
            M = Giac.combination(d, [λ])
            @test iszero(C.expand(M[1] * (x + y) + M[2] * x * y - (x^3 + y^3)))
        end
        @test decompose(x^3 + y^3, [x * y]; degree = 1) === nothing              # none of this degree
    end

    @testset "rational functions" begin
        d = decompose(1 / (x * (x + 1)), [1 / x, 1 / (x + 1)]; degree = 0)
        @test string.(d.multipliers) == ["1", "-1"]
    end

    @testset "a relation by a parametrization: the star identity" begin
        @giac_var τ u u′ l′ l″ t₁ t₂ t₃ A A′ A₃ g
        P, Q, e, dd = (τ - u) * (u′ - τ), (l′ - τ) * (l″ - τ), l″ - l′, u′ - u
        α = A′ * (t₃ - t₁) * (t₂ - τ) + A₃ * (t₂ - t₁) * (t₃ - τ) - A * (t₃ - t₂) * (τ - t₁)
        β = A * A′ * (t₂ - t₁) * (t₃ - τ) - A′ * A₃ * (t₃ - t₂) * (τ - t₁) + A * A₃ * (t₃ - t₁) * (t₂ - τ)
        δ₊, δ₋ = u′ - t₂, t₂ - l′
        T = A′ * e * α * P + dd * β * Q
        W = A′ * α * (τ - u) * ((t₂ - τ) * δ₋ + e * δ₊)
        V = β * (l″ - τ) * δ₊ * (τ - u / g)
        d = decompose(T, [:S => (t₂ - τ)^2, :W => W, :V => V]; degree = [5, 0, 0], param = (l′ => t₂ / g, u′ => t₂ * g))
        @test isempty(d.directions) && string.(d.multipliers[2:3]) == ["1", "-1"]
        M = d.multipliers[1]
        @test Bool(positivity(Giac.substitute(M, τ => t₁); order = (u, t₁, t₂, t₃, l″), positive = (A, A′, A₃)))
        @test Bool(positivity(Giac.substitute(M, τ => t₃); order = (u, t₁, t₂, t₃, l″), positive = (A, A′, A₃)))
    end
end

using JuMP, HiGHS

@testset "certify a member of the family" begin
    C = Giac.Commands
    @giac_var x y
    d = decompose(x^3 + y^3, [x + y, x * y]; degree = [2, 1])
    c = certify(d; positive = (x, y), signs = (1, -1))
    @test c !== nothing && all(r -> Bool(r), c.checks)
    @test iszero(C.expand(c.multipliers[1] * (x + y) + c.multipliers[2] * x * y - (x^3 + y^3)))
    @test certify(d; positive = (x, y), signs = (1, 1)) === nothing          # no member has both ≥ 0
end

@testset "a search over products of building blocks" begin
    @giac_var x y
    # (x + y)(x - y)² from the blocks x, y, (x - y)²: the products with nonnegative weights
    d = decompose(x^3 - x^2 * y - x * y^2 + y^3, [:x => x, :y => y, :s => (x - y)^2]; products = 3, degree = 0)
    @test length(d.blocks) == 19 && !isempty(d.directions)
    c = certify(d; positive = (x, y), block_signs = (1, 1, 1))
    @test Dict(n => string(m) for (n, m) in zip(d.names, c.multipliers) if !iszero(m)) == Dict("x·s" => "1", "y·s" => "1")
end
