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

# built and called in one function: no newer world needed, and the result type is inferred
_build_and_call(e, x, v) = build_function(e, x; backend = :julia)(v)

@testset "compiled functions: common subexpressions, called directly" begin
    @giac_var x y
    @test _build_and_call(x^2 + 1, x, 3.0) == 10.0
    f = build_function(x^2 + 1, x; backend = :julia)
    @test @inferred(f(2.0)) == 5.0

    # sin(x*y) is computed once, into a local variable, for both entries
    g = build_function([sin(x * y) + x, sin(x * y)^2], x, y; backend = :julia)
    @test g(0.5, 2.0) ≈ [sin(1.0) + 0.5, sin(1.0)^2]
    @test count(ex -> ex isa Expr && ex.head === :(=), g.code.args[2].args) == 1

    codes = Any[:(sin(a) + cos(a)), :(sin(a) * b)]
    assignments, outs = Giac._common_subexpressions(codes; taken = Set([:s1]))
    @test assignments == [:(s2 = sin(a))]                  # s1 is taken
    @test outs == Any[:(s2 + cos(a)), :(s2 * b)]
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
    @test endswith(T(Giac.Commands.series(exp(t), t, 0, 3)), " + O\\left(t^{4}\\right)")
    @test endswith(T(Giac.Commands.series(exp(t), t, 1, 2)), " + O\\left(\\left(t - 1\\right)^{3}\\right)")
    @test !occursin("order", T(Giac.Commands.series(sin(x), x, 0, 4)))
end

@testset "desolve with a function declared as y(t)" begin
    @giac_var t
    @giac_var y(t)
    eq = diff(y, t, 2) + 3diff(y, t) + 2y ~ 0
    ics = [y(0) ~ 1, Differential(t)(y)(0) ~ 0]
    @test string(Giac.Commands.desolve(eq, t, y)) == "c_0*exp(-t)+c_1*exp(-2*t)"
    @test string(Giac.Commands.desolve([eq; ics], t, y)) == "2*exp(-t)-exp(-2*t)"
    @test string(Giac.Commands.desolve([eq; ics], y)) == "2*exp(-t)-exp(-2*t)"     # t from y(t)
    # an Euler–Cauchy equation GIAC gives up on is solved in ln t
    ec = t^2 * diff(y, t, 2) - 3t * diff(y, t) + 4y ~ t^2 * log(t)
    s = Giac.Commands.desolve(ec, y)
    @test string(Giac.Commands.simplify(t^2 * diff(s, t, 2) - 3t * diff(s, t) + 4s - t^2 * log(t))) == "0"
    @test occursin("c_0", string(s)) && occursin("c_1", string(s))
end

@testset "declared assumptions" begin
    @giac_var k::positive n::integer x::real y::(0, 1) z
    @test string(sqrt(k^2)) == "k" && string(sign(k)) == "1"
    @test string(sin(n * pi)) == "0"
    @test string(conj(x)) == "x"
    @test string(sqrt(y^2)) == "y"
    @test string(sqrt(z^2)) == "abs(z)"                       # no tag, no assumption
    @giac_var k                                               # a new declaration starts afresh
    @test string(sqrt(k^2)) == "abs(k)"
    @giac_var w[1:2]::positive
    @test string(sqrt(w1^2) + sqrt(w2^2)) == "w1+w2"
    @giac_matrix M[2, 2]::positive
    @test string(sqrt(m12^2)) == "m12"
    # `assuming` adds to a declaration for its block, then restores it
    @giac_var r::real
    @test string(assuming(() -> sqrt(r^2), r > 0)) == "r"
    @test string(sqrt(r^2)) == "abs(r)" && string(conj(r)) == "r"
    @test_throws ArgumentError Giac._declare!(z, :wobbly)
    @test_throws ArgumentError Giac._process_giac_var_arg(:(u(t)::positive))   # functions take none
end

@testset "a complex name keeps its conjugate" begin
    @giac_var conj_c::complex
    @test string(conj(conj_c)) == "conj(conj_c)"
    @test Giac._declared(conj_c) === :complex
end

# A global declaration, made from any thread, is seen from every thread: GIAC keeps its state per
# OS thread, and its calls all run on one (see `with_giac_lock`)
@testset "a global declaration is seen from every thread" begin
    v = Giac.giac_identifier("gl_everywhere")
    fetch(Threads.@spawn Giac._declare!(v, :positive; scope = :global))
    seen = [fetch(Threads.@spawn string(Giac.giac_eval("about(gl_everywhere)"))) for _ in 1:8]
    @test all(==("assume[[],[line[0,+infinity]],[0]]"), seen)
    Giac._declare!(v, nothing; scope = :global)
end

# Declarations in a local scope belong to the task that ran them
@testset "local declarations are the declaring task's own" begin
    probe(x) = (string(sqrt(x^2)), string(Giac.giac_eval("about($x)")))
    to_a, to_b, log = Channel{Nothing}(1), Channel{Nothing}(1), Channel{Any}(4)
    ta = Threads.@spawn begin
        @giac_var lo_a::positive
        put!(log, (:a, probe(lo_a))); put!(to_b, nothing); take!(to_a)
        put!(log, (:a_again, probe(lo_a)))
    end
    tb = Threads.@spawn begin
        take!(to_b)
        @giac_var lo_a::negative
        put!(log, (:b, probe(lo_a))); put!(to_a, nothing)
    end
    wait(ta); wait(tb); close(log)
    r = Dict(collect(log))
    pos, neg = "assume[[],[line[0,+infinity]],[0]]", "assume[[],[line[-infinity,0]],[0]]"
    @test r[:a] == ("lo_a", pos)
    @test r[:b] == ("-lo_a", neg)
    @test r[:a_again] == ("lo_a", pos)             # B's declaration didn't reach A's
    # neither is the global state of the name, which has no assumption
    @test string(Giac.giac_eval("about(lo_a)")) == "lo_a" && Giac._declared_tag("lo_a") === nothing
    # a declaration at top level is global, and one in a function or `let` local
    @test Giac._binding_scope(@__MODULE__, :never_declared_here, 1) === :local
end
