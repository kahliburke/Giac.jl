# Ideals of Giac expressions decomposed in Oscar. Oscar is not in the test environment (it pins an
# older AlgebraicSolving); run this file in an environment with Giac and Oscar.

using Oscar

@testset "Oscar: ideal decomposition" begin
    @giac_var x y μ a
    # two branches of equilibria, crossing at the pitchfork
    @test Set(Set(string.(c)) for c in minimal_primes([μ * x - x^3])) == Set([Set(["x"]), Set(["x^2-μ"])])
    @test Set(Set(string.(c)) for c in minimal_primes([x^2 * y ~ 0, x * y^2 ~ 0])) == Set([Set(["x"]), Set(["y"])])
    pd = primary_decomposition([x^2, x * y])
    @test length(pd) == 2 && all(c -> c.prime isa Vector{GiacExpr}, pd)
    @test Set(Set(string.(c.prime)) for c in pd) == Set([Set(["x"]), Set(["x", "y"])])
    # generators are reduced: the Lorenz equilibria away from the origin, z = ρ - 1 and y = x
    @giac_var y z σ ρ β
    lorenz = [σ * (y - x), x * (ρ - z) - y, x * y - β * z]
    @test Set(Set(string.(c)) for c in minimal_primes(saturation(lorenz, σ * β))) ==
          Set([Set(["z-ρ+1", "x-y", "y^2-ρ*β+β"]), Set(["x", "y", "z"])])
    @test string.(radical([x^2 * y^2])) == ["x*y"]
    @test Set(string.(saturation([a * x, a * y - x], a))) == Set(["x", "y"])
    # calls from parallel tasks take turns in Singular
    tasks = [Threads.@spawn length(minimal_primes([μ * x - x^3])) for _ in 1:16]
    @test all(==(2), fetch.(tasks))
end
