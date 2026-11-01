# Polynomials in several variables and rational functions, between Giac and Nemo

using Nemo
import Nemo: ZZ, QQ
using Giac: to_giac, to_nemo

@testset "Nemo bridge: several variables" begin
    C = Giac.Commands
    @giac_var x y z t m x1 x2 v w

    @testset "into a polynomial ring" begin
        R, (X, Y, Z) = polynomial_ring(QQ, [:x, :y, :z])
        @test to_nemo((x + y)^2 / 2 - z, R) == (X + Y)^2 // 2 - Z
        @test to_nemo(C.expand((x - 2y + z)^5), R) == (X - 2Y + Z)^5
        @test to_nemo(convert(GiacExpr, 3), R) == R(3)
        S, (XS, YS) = polynomial_ring(ZZ, [:x, :y])
        @test to_nemo(x^2 * y - 3, S) == XS^2 * YS - 3
        @test_throws ErrorException to_nemo(x / 2, S)               # not an integer
        @test_throws ErrorException to_nemo(1 / x, R)               # a negative power
        @test_throws ErrorException to_nemo(sin(x), R)
        @test_throws ErrorException to_nemo(t * x, R)               # t is not a variable of R
        @test_throws ErrorException to_nemo(x * 1.5, R)
    end

    @testset "into a fraction field" begin
        R, (X, Y) = polynomial_ring(QQ, [:x, :y])
        F = fraction_field(R)
        @test to_nemo(1 / (x - y) + x, F) == F(1) // (F(X) - F(Y)) + F(X)
        @test to_nemo(x^-2 * y, F) == F(Y) // F(X)^2
    end

    @testset "a ring from the expressions" begin
        p, q = to_nemo((x^2 - y^2, x + y))
        @test parent(p) === parent(q) && string.(symbols(parent(p))) == ["x", "y"]
        @test divexact(p, q) == gens(parent(p))[1] - gens(parent(p))[2]
        r = to_nemo(1 / (x - y))
        @test parent(r) isa Nemo.FracField
        @test base_ring(parent(to_nemo(x + 1; base = ZZ))) === ZZ
        D = C.discriminant(m * (t - v) * (t - w) - (t - x1) * (t - x2), t)
        d = to_nemo(D)
        @test length(d) == 12 && Set(string.(symbols(parent(d)))) == Set(["m", "x1", "x2", "v", "w"])
    end

    @testset "back to Giac, names kept" begin
        R, (X, Y) = polynomial_ring(QQ, [:x, :y])
        g = to_giac(3X^2 * Y - Y // 2 + 1)
        @test g isa GiacExpr && iszero(C.expand(g - (3x^2 * y - y / 2 + 1)))
        @test iszero(C.normal(to_giac((X + 1) // (Y - X)) - (x + 1) / (y - x)))
        @test to_giac(zero(R)) == convert(GiacExpr, 0)
        p = to_nemo(C.expand((x + 2y - 3z + 1)^6))
        @test to_nemo(to_giac(p), parent(p)) == p                  # a round trip
    end
end
