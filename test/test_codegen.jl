# Equality and hashing, substitution with arrays, and Julia code from Giac expressions

@testset "isequal agrees with hash" begin
    @giac_var x y
    @test x + 1 == 1 + x                                   # mathematical equality
    @test sin(x)^2 + cos(x)^2 == 1
    a, b = x + 1, x + 1
    @test isequal(a, b) && hash(a) == hash(b)
    @test isequal(x + 1, 1 + x) == (hash(x + 1) == hash(1 + x))
    @test !isequal(sin(x)^2 + cos(x)^2, giac_eval("1"))  # structural: different expressions
    @test length(unique([x + 1, x + 1, y])) == 2
    @test Dict(x + y => 1)[x + y] == 1
end

@testset "substitute with arrays" begin
    @giac_var x
    @giac_var q[1:2]
    @test string(substitute(q1 + q2^2, q => [1, 2])) == "5"
    @test string(substitute(q1 + x, q => [1, 2], x => 5)) == "6"
    @test string((q1 * q2)(q => [3, 4])) == "12"
    A = GiacExpr[0 1; -2 -3]
    @test string.(substitute(A * q, q => [1, 1])) == ["1", "-5"]
    @test_throws DimensionMismatch substitute(q1, q => [1, 2, 3])
    @test string(substitute(sin(q1) + q2, q => (π, 0))) == "0"        # a tuple keeps π exact
    @test string(substitute(q1 + q2, q => [π, 0])) != "pi"             # the array literal is Float64
end

@testset "to_expr" begin
    @giac_var x y k
    T = Giac.to_expr
    @test T(x^2 + 1) == :(x ^ 2 + 1)
    @test T(sin(x) / y) == :(sin(x) / y)
    @test T(-x - y * k / 2) == :(-x - (y * k) / 2)
    @test T(3x / 4) == :((3x) / 4)
    @test T(giac_eval("pi") * x) == :(π * x)
    @test T(Giac.Commands.ln(x)) == :(log(x))
    @test T(giac_eval("1/3")) == 1 // 3
    @test T(giac_eval("f(x, 2)")) == :(f(x, 2))
    @test_throws ArgumentError T(giac_eval("erf(x)"))
end

@testset "build_function: arrays and the :julia backend" begin
    @giac_var x y u
    @giac_var q[1:2]
    F = [q2, -sin(q1) - q2 + u]
    f = build_function(F, q, u; backend = :julia)
    @test f isa Giac.CompiledFunction
    @test f([0.1, 0.0], 0.5) ≈ [0.0, 0.5 - sin(0.1)]
    @test build_function(x^2 + 1, x; backend = :julia)(3.0) == 10.0
    @test build_function([x y; 1 x * y], x, y; backend = :julia)(2, 3) == [2 3; 1 6]
    g = build_function(F, q, u)                            # :giac backend, same answers
    @test g([0.1, 0.0], 0.5) ≈ f([0.1, 0.0], 0.5)
    @test build_function([x y; 1 x * y], x, y)(2, 3) == [2 3; 1 6]
    @test_throws ArgumentError build_function(x + y, x; backend = :julia)   # y is not an argument
    @test_throws ArgumentError build_function([x], x; backend = :nope)
end

@testset "array ~ scalar" begin
    @giac_var q[1:2]
    G = [q1 * (1 - q1 - q2), q2 * (2q1 - 1)]
    eqs = G ~ 0
    @test eqs isa Vector{GiacExpr} && length(eqs) == 2
    @test length(solve(G ~ 0, q)) == 3                     # three equilibria
end

@testset "assuming" begin
    @giac_var k m
    @test string(assuming(() -> sqrt(k^2), k > 0)) == "k"
    @test string(sqrt(k^2)) == "abs(k)"                      # removed afterwards
    @test string(assuming(k > 0, k < 1) do
        sqrt(k^2)
    end) == "k"
    @test string(assuming(() -> sqrt(k^2) + sqrt(m^2), k > 0, m > 0)) == "k+m"
    @test_throws ErrorException assuming(k > 0) do
        error("raised inside")
    end
    @test string(sqrt(k^2)) == "abs(k)"                      # removed after an error too
end

@testset "series remainder typeset as O" begin
    @giac_var t x
    T(e) = Giac._entry_latex(e)
    @test endswith(T(Giac.Commands.series(exp(t), t, 0, 3)), "+O\\left(t^{4}\\right)")
    @test endswith(T(Giac.Commands.series(exp(t), t, 1, 2)), "+O\\left(\\left(t-1\\right)^{3}\\right)")
    @test !occursin("order", T(Giac.Commands.series(sin(x), x, 0, 4)))
end
