# Forms the core settles on: exponentials merged, the sign of an even or odd function's argument,
# multiples of π/2 taken out of cos and sin, simplify without new angles, exact literals

@testset "Normal forms" begin
    C = Giac.Commands
    L = Giac.giac_latex
    @giac_var x y u A::real B::real θ::real t::real E₁ E₂

    @testset "exponentials" begin
        @test isequal(exp(im * x) * exp(-im * x), convert(GiacExpr, 1))
        @test string(exp(x) * exp(y)) == "exp(x+y)" && string(exp(x)^2) == "exp(2*x)"
        @test string(exp(x) * exp(y) * exp(-x)) == "exp(y)"
    end

    @testset "the sign of an argument" begin
        @test isequal(cos(E₂ * t - E₁ * t), cos(E₁ * t - E₂ * t))      # even: one of the two
        @test L(sin(-x - y)) == "-\\sin\\left(x + y\\right)"
        @test L(sin(x - y)) == "\\sin\\left(x - y\\right)"              # odd, as many of each: as written
    end

    @testset "multiples of π/2" begin
        Π = convert(GiacExpr, π)
        @test string(cos(u + Π)) == "-cos(u)" && string(sin(u + Π / 2)) == "cos(u)"
        @test string(cos(Π / 2 - u)) == "sin(u)" && string(sin(u + 3Π / 2)) == "-cos(u)"
    end

    @testset "GIAC is given the normal form" begin
        z = A * exp(im * θ) + B * exp(-im * θ)
        ρ = C.expand(z * conj(z))
        @test isequal(C.expand(C.exp2trig(ρ)), A^2 + B^2 + 2A * B * cos(2θ))
    end

    @testset "simplify without new angles" begin
        @test string(C.simplify(cos(x)^2 * sin(y) + sin(x)^2 * sin(y))) == "sin(y)"
        @test string(C.simplify(sin(x) / (1 + cos(x)))) == "tan(1/2*x)"        # shorter: taken
        # the position in a superposition of two states of a well: cos θ, not a cubic in cos(θ/3)
        @giac_var a::positive ħ::positive m::positive
        Π = convert(GiacExpr, π)
        φ(n) = sqrt(2 / a) * sin(n * Π * x / a)
        ω(n) = n^2 * Π^2 * ħ / (2m * a^2)
        Ψ = (φ(1) * exp(-im * ω(1) * t) + φ(2) * exp(-im * ω(2) * t)) / sqrt(convert(GiacExpr, 2))
        X = C.simplify(C.exp2trig(C.integrate(x * C.expand(Ψ * conj(Ψ)), x, 0, a)))
        @test string(X) == "(9*a*pi^2-32*a*cos(3*pi^2*t*ħ/(2*a^2*m)))/(18*pi^2)"
        @giac_var a ħ m
    end

    @testset "expanding polynomials" begin
        e = C.expand((1 + x + y + u)^20)
        @test length(Giac.arguments(e)) == 1771
        @test string(C.expand((x // 2 + 1 // 3)^3)) == "1/8*x^3+1/4*x^2+1/6*x+1/27"
        @test isequal(C.expand((x + 1)^2 - 1), x^2 + 2x)
        @test string(C.expand((x + 1)^2 / (y + 1))) == "x^2/(y+1)+2*x/(y+1)+1/(y+1)"   # not a polynomial: over each term
        @test !Giac._is_polynomial(Giac._node((x + y)^2 * sin(x))) && !Giac._is_polynomial(Giac._node((1.5x + 1)^2))
    end

    @testset "exact literals" begin
        ex = Giac._exact_literals(:(sin(2π * x) + sqrt(2) * x^(1 / 2) + 2^(1 // 3) * ℯ))
        @test string(Base.invokelatest(eval(:(x -> $ex)), x)) == string(sin(2 * convert(GiacExpr, π) * x) + sqrt(convert(GiacExpr, 2)) * x^(1 // 2) +
                                          convert(GiacExpr, 2)^(1 // 3) * convert(GiacExpr, ℯ))
        @test Giac._exact_literals(:(f(1.5, 2 * 3))) == :(f(1.5, 2 * 3))           # floats and integers as written
        @giac_var a::positive
        @test string(@exact sqrt(2 / a) * sin(2π * x / a)) == string(sqrt(2 // 1 / a) * sin(2 * convert(GiacExpr, π) * x / a))
        @test !occursin(".", string(@exact 2π * x + sqrt(2)))
        @giac_var a
    end
end
